# Refine Pipeline Refactor Plan（自适应拆分 + 集中化 PDF→LLM 调用）

> 目标来源：`.claude/refine-pipeline-refactor.md`（handoff）
>
> 本文是“把计划写出来”的版本：不讨论代码细节实现到每一行，但把**要改什么、改在哪、验收标准**写清楚，便于按图施工、后续补丁更轻松。

---

## 0. 背景与结论（来自 handoff）

- 已确认：Google/Gemini 的 `503 UNAVAILABLE` 在大 PDF 上主要是**页数阈值**问题，而不是 JBIG2/PNG 等格式问题。
- 当前 refine pipeline 的批大小固定（`PdfBatchContext.batch_size=900`），一旦 503，代码多处会：
  - 直接重试同样请求，或
  - 走“rasterize/JBIG2 fallback”
  - 但这两者都不能从根本上解决“页数过多导致 503”。
- 必要改动：所有 “PDF → Google API（或任何会 503 的 PDF 输入接口）” 的调用都需要统一实现：
  1) **自适应页数缩小**：503 → 页数折半（450→225→…）直到成功或到最小阈值  
  2) **子批结果合并**：调用方拿到的是一个“合并后的结果”，不需要懂拆分细节  
  3) **会话内学习（learned limit）**：一旦发现 300 页可用，不要再试 900

---

## 1. 仓库现状（与 refine 相关的关键模块）

### 1.1 入口与主流程

- `pdf2epub/cli.py`：`refine` 子命令入口。
- `pdf2epub/refine/main.py`：`RefinedBreakdown.process()` 主编排：
  - 依赖 `preprocess_pdf()` 生成 `output/<title>/input.pdf`（页码 patch + 可能压缩）
  - 调 `StructureAnalyzer.analyze_pdf_structure()` 产出初始 TOC tree（以及 metadata）
  - 再进入 boundary verification agent（`boundary_agent.py`）做边界修订

### 1.2 PDF 预处理与压缩

- `pdf2epub/utils/pdf_utils.py`：
  - `add_page_number_patches()`：给每页打上 “PDF Page: X” 白色补丁（避免 printed page number 误导）
  - `preprocess_pdf()`：保存 `input_original.pdf`，生成 `input.pdf`；当文件 >45MB 时做压缩（JBIG2→PNG fallback）
- `pdf2epub/refine/structure_analyzer.py`：
  - `analyze_pdf_structure()` Step 0：当 payload > `refine.pdf_compression.payload_limit_mb`（默认 30MB）再压一次，并设置 `_pdf_already_rasterized`。
  - `_prepare_pdf()`：对 include/exclude 页做 subset，并在 subset > payload limit 时再做压缩。

> 结论：**payload 压缩**与**页数拆分**是两条正交路径；现有压缩逻辑可以保留，但 503 的正确处理应该优先“减页数”，而不是“换压缩格式”。

### 1.3 503 处理散落位置（需要集中化）

在 `pdf2epub/refine/structure_analyzer.py` 中，至少以下方法都各自写了 try/except + 503 fallback：

- `detect_toc_location()` / `_detect_toc_location_rasterized()`
- `extract_toc_structure()`
- `match_toc_with_content()` / `_match_toc_batched()`
- `_analyze_pdf_directly()` / `_analyze_pdf_batched()`

问题：

- 503 分支逻辑重复，且主要 fallback（rasterize）并不解决页数阈值。
- “批处理逻辑”（拆页/overlap/合并）与 “API 调用逻辑”（准备 PDF bytes、请求、parse、retry）交织，导致：
  - 改一次策略需要改 6+ 处
  - 很难写单元测试验证拆分/合并行为

---

## 2. 重构目标（明确可验收）

### 2.1 功能目标

1. **所有 PDF→LLM（带 PDF bytes 的请求）统一走一个入口**：集中处理
   - PDF subset + payload 压缩
   - 调用 client.generate_content_stream
   - JSON parse 与基础校验
   - 503 → adaptive split（折半）→ merge
2. **会话级 learned page limit**
   - 第一次 503 触发后，限制会降低并对后续所有 PDF→LLM 调用生效（同 provider/model 或同类操作维度可配置）
3. **调用方只关心业务结果，不关心拆分细节**
   - 例如 `_match_toc_batched()` 不再自己 try/except 503；它只提供：
     - 如何生成 prompt（含 batch range 信息）
     - 如何合并 batch 的 `chapters_found`

### 2.2 工程目标（让以后补丁更轻松）

- 503/拆分/合并逻辑只有一个地方改：减少 copy-paste。
- 关键组件可单测（不依赖真实 API）：
  - “当页数>阈值抛 503 时，是否折半直到成功”
  - “learned limit 是否生效”
  - “合并结果是否稳定（顺序、去重、边界）”
- 日志与可观测性清晰：
  - 每次 PDF→LLM 调用打印：页数、payload MB（如可得）、当前 learned limit、拆分次数

---

## 3. 目标架构（建议）

> 名称仅建议，实际以代码风格为准。

### 3.1 新增三个核心抽象

1. `PdfPageLimitLearner`
   - 维护会话内 `max_pages_per_request`
   - API：
     - `get_limit(operation_key) -> int`
     - `report_503(operation_key, attempted_pages) -> None`（通常折半）
     - `report_success(operation_key, attempted_pages) -> None`（可选：提升/稳定策略，默认不提升更稳）
   - `operation_key` 建议至少包含：`provider + model + operation_name`（可简化为 provider/model 级别）

2. `PdfRequestSpec`（或同等结构体）
   - 描述一次“带 PDF”请求需要的所有信息：
     - `pdf_path`
     - `pages`（include list）或 `exclude_pages`
     - `payload_limit_mb` / 是否允许压缩
     - `prompt_builder(pages_subset_meta) -> prompt`
     - `client/model/generation_config`
     - `parse_fn(text) -> dict`
     - `merge_fn(list[dict]) -> dict`（当拆成多次调用时）

3. `adaptive_pdf_call()` orchestrator
   - 统一入口：
     - 生成 page batches（尊重 learner limit + overlap 策略）
     - 对每个 batch：prepare bytes → call → parse
     - 503：触发 learner 降低 limit，并对当前 batch 进行二分/重排后重试
     - 汇总 batch results → merge → 返回

### 3.2 合并策略（按操作分类）

> 这里是关键：不同调用的“合并”含义不同，必须显式定义 merge_fn，避免隐式逻辑散落。

- **TOC location detection**（返回 `{has_toc, toc_start, toc_end}`）
  - merge：如果任一子结果 `has_toc=true`，选择最可信的一条：
    - 优先 `toc_start` 更小者（更靠前 TOC 更常见）
    - 或者按置信度/规则（若后续加字段）
- **TOC structure extraction**（返回纯结构：`chapters[]` 无页码）
  - 通常页数很少，不应触发 503；可先不实现复杂 merge（只要支持折半后“取第一份成功结果”也行）
  - 若确实需要 merge：需更复杂的树合并（风险高，建议最后做）
- **TOC matching batch scan（Phase 1）**（返回 `chapters_found: [{title,start_page}]`）
  - merge：concat 后去重（按 title 归一化），保留更早 `start_page`
- **Direct analysis batch scan**（返回 `chapters`（树或扁平）+ metadata（仅首/末批））
  - merge：
    - metadata：取 first batch 的 author/language/...，last batch 的 back_cover
    - chapters：concat + `deduplicate_chapters()`（已存在）

---

## 4. 详细执行计划（同步自对话 plan，并补充验收点）

> 下面步骤按“先铺基础设施，再逐步迁移调用点，最后清理/补文档/加测试”的顺序。

### Step 1 — 盘点所有 PDF→LLM 调用点（Inventory）

- 目标：列清楚所有“传 PDF bytes”的调用，标注：
  - operation_name
  - 使用的 model/provider（structure_model / toc_model）
  - pages 输入模式（full / include / exclude / batch）
  - 返回结构与合并需求
- 产物：在本文末尾维护一张表（或另起附录）。
- 验收：能指出每一处 503 try/except 的位置与差异。

### Step 2 — 复用/对齐现有 retry、split 思路（Review patterns）

- 目标：明确哪些重试应由底层 client 负责（网络抖动/429 等），哪些必须由 orchestrator 负责（503 页数阈值）。
- 对齐点：
  - `pdf2epub/utils/retry_utils.py`：通用 transient retry
  - `pdf2epub/utils/llm_client.py`：provider 抽象 + streaming
- 验收：定义清晰的“503 page limit”与“transient 503”区分策略（至少先按当前经验：PDF 503 视为页数问题优先处理）。

### Step 3 — 写清当前 503 重复与错误 fallback（Document duplication）

- 目标：在设计层面确认“rasterize fallback”不再是 503 的主路径。
- 产物：在本文记录：
  - 哪些方法 currently rasterize on 503
  - 为什么应该改为 adaptive split
  - rasterize 的合理定位：payload 压缩/兼容性（非页数阈值）
- 验收：重构后 structure_analyzer 中不再出现 6+ 处 copy-paste 503 处理（最终状态）。

### Step 4 — 设计 learned limit 的状态与粒度（Learner state）

- 决策点：
  - learner 的 key：`provider+model` 还是 `provider+model+operation`？
  - min_limit：例如 50/100 页，低于则直接失败并提示“该 PDF 在当前 API 条件下不可用”
  - 是否允许“成功后升回去”：建议默认不升（避免抖动），或只在多次成功后小幅提升
- 验收：同一次 refine session 内，一旦发现可用上限，后续不会再尝试更大的页数。

### Step 5 — 定义统一的 PDF 请求/响应抽象（Abstractions）

- 目标：把调用方需要提供的内容限制为：
  - prompt（可能依赖 batch range 元信息）
  - parse_fn（JSON）
  - merge_fn（当拆成多次调用）
- 验收：调用点迁移时“改动量小”，后续新增 PDF→LLM 操作只需写 prompt+merge。

### Step 6 — 实现页列表拆分工具（Splitting utility）

- 目标：支持两类 pages：
  - 连续页（content batches）
  - 非连续页（TOC detection：first+last）
- overlap 策略：
  - 对连续页：保留小 overlap（已有 `PdfBatchContext.overlap`）
  - 对非连续页：不 overlap，直接按列表切分
- 验收：可在单测里验证：给定 pages 和 limit，输出 batches 的大小与覆盖正确。

### Step 7 — 集中化 PDF subset + payload 压缩（PDF preparation）

- 目标：`_prepare_pdf()` 成为唯一的“从 pdf_path + pages → bytes”的实现点。
- 改造点（只列计划）：
  - 让 orchestrator 调用 `_prepare_pdf()`，而不是每个业务方法自己去 open/read。
  - 记录/输出更一致的日志：subset 页数 + MB + 是否压缩。
- 验收：调用点不再各自 `open(pdf_path,'rb')` 与临时压缩分支。

### Step 8 — 集中化 LLM 调用 + JSON parse（Call + parse）

- 目标：把 `generate_content_stream` + `parse_llm_json` 打包为一层：
  - 统一 operation_name
  - 统一 json_mode config
  - 统一异常信息（为 learner 提供 attempted_pages）
- 验收：在单测中可通过 fake client 注入异常/返回值。

### Step 9 — 实现 503 adaptive split-and-merge orchestrator（核心）

- 核心行为：
  - 初始 batch_size 使用：
    - `PdfBatchContext.batch_size`（默认 900）或
    - learner 当前 limit（若已学习）
  - 若某 batch 触发 503：
    - learner 降低 limit（通常折半）
    - 当前 batch 立即按新 limit 拆分重试
  - 成功后收集结果，最后 merge。
- 验收：
  - 当阈值=300 时，900 页 batch 会拆为 450→225 并成功
  - learned limit 生效：后续 batch 直接用 225/300，不再试 900

### Step 10 — 迁移 TOC location detection（detect_toc_location）

- 目标：`detect_toc_location()` 不再手写 503 try/except。
- 注意：TOC detection pages 是“first+last N”非连续列表：
  - 若触发 503，拆分为两个列表分别检测，merge 结果。
- 验收：大 PDF（>1000 页）在 503 条件下仍能完成 TOC detection。

### Step 11 — 迁移 TOC structure extraction（extract_toc_structure）

- 目标：同样走统一入口；但预计不会触发 503（TOC 页数很少）。
- 验收：逻辑更简单，错误处理一致。

### Step 12 — 迁移 TOC matching Phase 1 扫描（_match_toc_batched）

- 目标：把 per-batch `try/except 503` 删除，交给 orchestrator：
  - prompt 仍保留 “Only report chapters START within this page range”
  - merge：concat `chapters_found` + 去重
- 验收：
  - 1084 页书在 learned limit ~300 时能稳定跑完 Phase 1
  - Phase 2（merge_prompt）保持不变（它不带 PDF）

### Step 13 — 迁移 direct analysis batched（_analyze_pdf_batched）

- 目标：同样用 orchestrator；metadata 仅从 first/last batch 提取。
- 验收：503 不再触发 rasterize fallback 的“无效重试”，而是自动拆分。

### Step 14 — 增加单元测试（Fake 503）

- 目标：不依赖真实 API，构造 fake client：
  - 当 pages_count > threshold：抛出包含 “503/unavailable” 的异常
  - 否则返回可解析 JSON（最小结构）
- 覆盖：
  - 折半逻辑
  - learned limit
  - merge_fn 行为（TOC detection、chapters_found concat）
- 验收：CI/本地 `pytest` 通过；测试能在 1s 量级跑完。

### Step 15 — 配置项、文档、日志（Knobs + docs）

- 新增/调整 config（建议）：
  - `refine.adaptive_page_limit.initial_pages`（默认 900）
  - `refine.adaptive_page_limit.min_pages`（默认 50/100）
  - `refine.adaptive_page_limit.overlap_pages`（默认沿用 50）
  - 是否按 provider/model 分 bucket 学习
- 日志要求：
  - 每次 503：打印 attempted_pages、new_limit、operation_name
  - 每次调用：打印 pages_count、payload_mb（如可得）、limit
- 验收：日志能解释“为什么会拆分到多少页”，便于后续打补丁。

### Step 16 — 清理死代码与 TODO（Cleanup）

- 重点：
  - 删除/收敛旧的 503 rasterize-only 分支（或至少不再作为默认路径）
  - 标记 `convert_toc_page_to_original()`：当前已不再使用（handoff 提到双转换 bug 已修复）
  - `compress_pdf_bytes` 未实现：记录为独立 issue（与本重构无关）
- 验收：结构分析相关代码的异常处理路径明显减少、可读性提升。

### Step 17（可选）— 将 learned limit 持久化到 state（Resume 更强）

- 动机：resume 时避免再次从 900 试起，节约时间与 API。
- 位置：`RefinerState`（`output/<title>/refiner_state.json`）
- 风险：不同时间/负载阈值可能变化；持久化需设置 TTL 或“失败后自动降低”即可自适应。
- 验收：resume 后直接使用上次 learned limit；若再次 503，会继续折半降低。

---

## 5. 附录：待盘点调用点表（Step 1 产物模板）

| 调用点 | 是否带 PDF | pages 模式 | 现在的 503 处理 | 目标 merge 策略 |
|---|---:|---|---|---|
| `detect_toc_location` | ✅ | include（first+last） | 503→rasterize | 任一 has_toc=true 则选最优 |
| `extract_toc_structure` | ✅ | include（TOC pages） | 503→rasterize |（通常无需拆分） |
| `match_toc_with_content` | ✅ | exclude（TOC pages） | 503→rasterize | 单次结果 |
| `_match_toc_batched` Phase 1 | ✅ | include（batch pages） | 503→rasterize | concat+去重 |
| `_analyze_pdf_directly` | ✅ | full | 503→rasterize | 单次结果 |
| `_analyze_pdf_batched` | ✅ | include（batch pages） | 503→rasterize | concat+dedupe |

