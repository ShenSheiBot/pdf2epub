# pdf2epub

将外语的学术书籍或者纵排日语书籍的扫描件转换成epub格式，保留完备的目录、注音、脚注、插图、表格（公式待支持）等信息，具备完备的链接跳转功能，使其尽可能接近出版社原epub的排版。

## 技术优势
经测试，本项目效果远强于各大商业软件的直接转换效果，同时因为其基于OCR的特性，不会因出版社更新DRM机制而失效。

### Demo
转换前：http://biopolitics.kom.uni.st/Michel%20Foucault/The%20Foucault%20Reader%20(149)/The%20Foucault%20Reader%20-%20Michel%20Foucault.pdf

转换后：https://raw.githubusercontent.com/ShenSheiBot/pdf2epub/refs/heads/main/example.epub

## 局限性
因为其复杂性和对多模态LLM的依赖，转换速度较慢并有小概率可能会因为LLM的审核原因失败。第一步的目录分解和术语表提取强制需求 gemini 的大 context。剩余步骤建议尽量避免 gemini（审核最严格）。

对于纵排日语，需要扫描文件的质量较高且为*白底*。（并非白底会导致插图识别错误）

因为逐页进行转换，要求书的新章节*新起一页*，否则章节的最后部分可能会被顺延到下一章节。如果扫描件是每两页一扫描的pdf，建议拆分成单页pdf再操作。

## 工作流程 (Workflow)

### PDF 工作流 (Recommended - uses toc_tree.json)

适用于 PDF 扫描件转换：

1. **ocr-pages**：逐页进行 OCR，保存 Markdown、HTML 和可审计的原始版式信息
2. **refine**：智能分析 TOC 结构，验证章节边界，生成精确的 toc_tree.json（支持无限层级嵌套）
3. **polish**：使用 LLM 建立正确的链接跳转，消除 OCR 错误、页眉页脚等
4. **translate**：（可选）使用 LLM 翻译成目标语言
5. **build-epub**：基于 toc_tree.json 生成 EPUB

### EPUB 翻译工作流 (NEW - 保留原始格式)

适用于已有 EPUB 文件的翻译，完整保留原书的 CSS 样式、字体、排版：

1. **translate-html**：直接翻译 EPUB 内的 XHTML 内容，保留所有 HTML 结构和样式
2. **build-html-epub**：将翻译后的 HTML 重新打包成 EPUB

```bash
# EPUB 翻译示例
uv run pdf2epub translate-html -i book.epub --target-language Chinese
uv run pdf2epub build-html-epub
```

优势：
- 完整保留原书的 CSS 样式、字体、封面、目录结构
- 翻译后的书籍排版与原书一致
- 支持增量翻译（`--resume`）和部分测试（`--limit N`）

### 轻小说翻译工作流 (Novel - 文本模式 + 术语表)

适用于轻小说 EPUB 的日→中翻译，支持术语表记忆和退化防护：

1. **translate-novel**：逐章翻译，自动提取/维护术语表，embedding验证对齐
2. **build-novel-epub**：从翻译文本重建 EPUB（支持部分翻译）

```bash
# 轻小说翻译示例
uv run pdf2epub translate-novel -i novel.epub -c config.yaml
uv run pdf2epub build-novel-epub -c config.yaml
```

优势：
- 跨章术语一致性（GlossaryManager 自动维护）
- Sonnet 退化防护（streaming guard + chunked fallback）
- Embedding-based 对齐验证（无安全过滤限制）
- DeepSeek 作为 fallback（R18 内容无审核）
- 支持 `--resume`、`--retranslate <chapter>`、`--limit N`

### arXiv / LaTeX 翻译工作流

适用于 arXiv 论文源码或本地 TeX 工程。它不经过 PDF OCR，而是直接翻译
TeX 正文，并将“完整工程可以用 XeLaTeX 编译”作为每个翻译单元的提交条件。

```bash
# 直接下载 arXiv 源码、翻译并重新编译
uv run pdf2epub translate-arxiv 2503.01800

# 翻译本地工程；入口也可以自动识别
uv run pdf2epub translate-arxiv ./paper-source --main-tex main.tex

# 只处理下一个单元，适合先做小规模验证
uv run pdf2epub translate-arxiv 2503.01800 --limit 1
```

该工作流会自动注入 `ctex`、递归跟踪正文中的 `\input` / `\include` /
`\subfile`，并在 `output/arxiv/<source-id>/project` 中留下可独立重新编译的
工程。状态、单元译文、编译日志和内容寻址缓存保存在同一运行目录的
`.pdf2epub` 下；重复执行同一命令会从已编译成功的单元继续，不会再次请求
LLM。只有候选译文无法编译时才调用 whole-mode repair agent；修复失败则保留
该单元原文。默认修复模型为 `gpt-5.6-luna`，并可通过配置中的
`type: codex` 复用本机 Codex 当前选中的 OpenAI-compatible provider，
无需把 bearer token 复制到 `config.yaml`。

需要本机安装包含 XeLaTeX、`latexmk`、`ctex` 和 Fandol 字体的 TeX Live。

## 推荐LLM：

- refine / entity extraction：gemini-2.5-pro
- polish / translate：claude-sonnet-4-6 或 deepseek-chat
- translate-novel：claude-sonnet-4-6（主）+ deepseek-chat（fallback）

## 日语OCR架构

### OCR后端支持

本项目支持以下 OCR 后端。对于日语纵排、英语、复杂表格和混合版型，当前
推荐使用 Chandra 2；传统后端继续保留用于兼容和对照。

#### 1. **Chandra 2** (`chandra`)
- 通过本地 vLLM 服务运行，单张 24 GB NVIDIA GPU 即可
- 同时保存 Markdown、保留 bbox/label 的 HTML、原始 HTML、ordered blocks 和裁图
- 支持日语纵排、英语、表格、公式、脚注、页眉页脚和混合版型
- 服务部署与重启说明见 [`deploy/chandra`](deploy/chandra/README.md)

#### 2. **Azure Document Intelligence** (`azure`)
- 支持振假名(furigana)检测和重组
- 需要Azure账户和API密钥

#### 3. **Google Cloud Vision** (`vision`)
- 正文偶有漏字，振假名偶有错漏
- 需要Google Cloud账户和服务账户密钥

#### 4. **通用 Vision Language Models** (`vllm`)
- Gemini 识别效果较佳，但经常“自由发挥”，添加不存在的振假名，且审核严格，不推荐
- Anthropic 识别效果较差，虽然审核宽松，更不推荐
- VLLM 整体识别速度缓慢且费用较高，胜在输出文本连贯，但仍不能完全摆脱后处理需求，故整体仅作为备用方案

## 安装

### 依赖要求
- Python 3.11+
- UV (包管理器)
- 至少一个OCR后端的API账户
- 构建含 LaTeX 公式的 PDF EPUB 时，需要包含 `xelatex`、`dvisvgm`、
  `xeCJK`、`mathtools` 和 Fandol 字体的 TeX Live

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/pdf2epub.git
cd pdf2epub
```

2. 安装 UV（如果未安装）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. 安装项目依赖
```bash
uv sync
```

4. 配置 API 密钥
```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml 填入你的 API 密钥
```

### 配置

参考 `config.yaml.example`。基本结构：

```yaml
title: "书名"

credentials:
  providers:
    anthropic:
      type: anthropic
      api_key: your-key
      base_url: https://api.anthropic.com  # 可选
    gemini:
      type: google
      api_key: your-key
    deepseek:
      type: openai
      api_key: your-key
      base_url: https://api.deepseek.com/v1
    # 可选：供 refine 的 OpenAI Responses PDF 传输使用。
    openai_pdf:
      type: openai
      api_key: your-key
      base_url: https://your-responses-compatible-endpoint.example/v1

ocr:
  backend: chandra  # chandra / azure / vision / vllm

translation:
  source_language: Japanese
  target_language: Chinese
  models:
    - provider: anthropic
      model: claude-sonnet-4-6
      api_retries: 2
      validation_retries: 2
    - provider: deepseek
      model: deepseek-chat
      api_retries: 2
      validation_retries: 2

# 可选：只在明确指定时让 refine 通过 OpenAI Responses API 传入 PDF。
# 这是 Gemini PDF Part 路径的备用方案，不会在失败时自动切换模型。
refine:
  structure:
    provider: openai_pdf
    model: "your-pdf-capable-model"
    toc_model: "your-pdf-capable-model"
    pdf_transport:
      type: openai_responses
      timeout_seconds: 600

# `base_url` 可写兼容端点根地址、.../v1，或 .../v1/responses；适配器会归一化。
# 分批页数和重叠页数取决于模型、端点和 PDF，不存在通用的 50 页上限。
# 明确的上下文超限会停止 refine 并保留错误，不会自行切换模型。

# 轻小说专用配置（可选）
novel:
  glossary_max_tokens: 1000
  embedding_provider: gemini       # embedding 验证
  embedding_model: gemini-embedding-001
```

### 2. 推荐工作流程（统一CLI）

所有功能通过统一的CLI入口访问：

#### 步骤 1: 页级 OCR
```bash
uv run pdf2epub ocr-pages -i input.pdf
```
生成 `output/{book_title}/pages/page_*.md`

参数说明：
- `--resume`: 从上次中断处继续
- `--start-page`: 起始页码
- `--end-page`: 结束页码
- `--max-workers`: 并发数

#### 步骤 2: 精细化拆分
```bash
uv run pdf2epub refine
```
分析 TOC 结构并验证章节边界，生成 `output/{book_title}/toc_tree.json`（支持无限层级嵌套）

参数说明：
- `--resume`: 从上次中断处继续
- `--max-tokens`: 每个单元的最大 token 数

#### 步骤 3: 内容润色
```bash
uv run pdf2epub polish
```

针对不同内容类型：
```bash
# 学术书籍（带脚注和引用）
uv run pdf2epub polish --content-type academic

# 日语书籍（保留振假名）
uv run pdf2epub polish --content-type japanese

# 自动检测内容类型
uv run pdf2epub polish --content-type auto
```

也可以把整本 Markdown 工作目录交给同一个 Codex 会话统一润色；该模式直接
使用 Codex CLI 自身已有的配置和进程环境，不把凭证写入本项目配置：

```bash
uv run pdf2epub polish --workspace-agent --content-type academic

# 中断后恢复同一个会话
uv run pdf2epub polish --workspace-agent --content-type academic --resume
```

#### 步骤 4: 生成 EPUB
```bash
uv run pdf2epub build-epub
```
最终 EPUB 文件保存在 `output/{book_title}/output.epub`

如需从翻译后的内容生成：
```bash
uv run pdf2epub build-epub --translated
```

### 3. 翻译功能

#### 实体提取（可选，用于保持翻译一致性）

对于包含大量专有名词的书籍（如日语轻小说），可以先提取实体：

```bash
# 提取人物、地点、术语等实体
uv run pdf2epub extract-entities -i input.pdf --source-lang Japanese --target-lang Chinese
```

生成 `output/{book_title}/translation_entities.json`，包含：
- **人物名称**：包含性别、描述、关系
- **地点名称**：城市、建筑、幻想世界
- **组织机构**：公会、学校、公司
- **专有术语**：魔法、技能、道具
- **种族物种**：包含单复数形式

#### 翻译处理

```bash
# 基本翻译（自动检测并使用实体文件，如果存在）
uv run pdf2epub translate --target-language Chinese

# 强制不使用实体（即使文件存在）
uv run pdf2epub translate --target-language Chinese --no-entities
```

**注意**：如果 `translation_entities.json` 文件存在，翻译器会自动使用它以保持一致性。

### 4. 完整工作流程示例

#### 日语轻小说翻译流程
```bash
# 1. 页级OCR
uv run pdf2epub ocr-pages -i manga.pdf

# 2. 精细化拆分
uv run pdf2epub refine

# 3. 提取翻译实体（可选，用于一致性）
uv run pdf2epub extract-entities

# 4. 日语内容润色
uv run pdf2epub polish --content-type japanese

# 5. 翻译成中文（自动使用已提取的实体）
uv run pdf2epub translate --target-language Chinese

# 6. 生成EPUB
uv run pdf2epub build-epub --translated
```

#### 学术书籍翻译流程
```bash
# 1. 页级OCR
uv run pdf2epub ocr-pages -i thesis.pdf

# 2. 精细化拆分
uv run pdf2epub refine

# 3. 学术内容润色（保留脚注）
uv run pdf2epub polish --content-type academic

# 4. 翻译
uv run pdf2epub translate --target-language Chinese

# 5. 生成EPUB
uv run pdf2epub build-epub --translated
```

#### 已有 EPUB 翻译（保留原格式）
```bash
uv run pdf2epub translate-html -i book.epub --target-language Chinese
uv run pdf2epub build-html-epub
```

#### 轻小说翻译（术语表 + 退化防护）
```bash
uv run pdf2epub -c config.yaml translate-novel -i novel.epub
# 中断后继续
uv run pdf2epub -c config.yaml translate-novel -i novel.epub --resume
# 单独重建 EPUB（不重翻）
uv run pdf2epub -c config.yaml build-novel-epub
```

#### 英文书籍（无需翻译）
```bash
# 1. 页级OCR
uv run pdf2epub ocr-pages -i book.pdf

# 2. 精细化拆分
uv run pdf2epub refine

# 3. 内容润色
uv run pdf2epub polish

# 4. 生成EPUB
uv run pdf2epub build-epub
```

### 5. 高级配置

系统支持在模型失败或触发安全审核时自动 fallback 到下一个 provider。在 `translation.models` 数组中配置多个 provider 即可。

### 6. 故障排除

#### OCR 失败
- 检查 API 配额和密钥配置
- 降低 `max_workers` 减少并发
- 使用 `--resume` 从失败处继续

#### 审核问题
- 配置多个模型提供商
- Gemini 被阻止时会自动切换到 Anthropic

#### 内存不足
- 减少 `max_workers`
- 降低 `zoom_factor`
- 分批处理章节

### 7. 输出结构
```
output/
└── {book_title}/
    ├── input.pdf              # 处理后的PDF
    ├── input_original.pdf     # 原始PDF副本
    ├── toc_tree.json          # TOC结构（支持无限层级）
    ├── pages/                 # 页级OCR结果
    │   ├── page_001.md
    │   ├── page_002.md
    │   └── ...
    ├── ocr_markdown/          # 聚合后的章节内容
    │   ├── chapter_1.md
    │   ├── chapter_1.1.md     # 支持层级嵌套
    │   └── ...
    ├── polished_markdown/     # 润色后内容
    │   ├── chapter_1.md
    │   └── ...
    ├── images/                # 提取的插图
    │   ├── ch001_p010_illustration.png
    │   └── ...
    ├── translated/            # 翻译后内容（如果执行了翻译）
    │   ├── chapter_1.md
    │   └── ...
    ├── translation_entities.json  # 翻译实体参考（如果提取了）
    ├── translation_reference.txt  # 人类可读的翻译参考
    ├── pages/ocr_progress.json           # 页级OCR进度
    ├── ocr_markdown/tree_progress.json   # refine进度
    ├── polished_markdown/processing_tracker.json   # 润色进度
    ├── translated/processing_tracker.json          # 翻译进度
    └── output.epub            # 最终 EPUB
```


## 贡献

欢迎提交Issue和Pull Request！
关注[甚谁](https://www.zhihu.com/people/sakuraayane_justice)谢谢喵！

## 许可

MIT License
