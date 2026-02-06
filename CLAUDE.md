# pdf2epub 项目使用指南

## 项目概述

将扫描版 PDF 或 EPUB 转换/翻译为高质量 EPUB 的工具，特别适合处理日文竖排书籍。

---

## 关键风险点和避雷指南

### 1. Output 目录结构 - 绝对不要随意删除

**结构**：`output/{title}/` 包含所有处理数据

```
output/{title}/
├── input.epub / input.pdf    # 输入文件副本
├── input_original.pdf        # PDF 原始备份（重要！）
├── pages/                    # 页级 OCR 结果
│   └── ocr_progress.json     # OCR 进度
├── compressed_units/         # EPUB HTML 压缩后内容
├── translated_compressed/    # 翻译后内容
├── polished_markdown/        # 润色后内容
│   └── processing_tracker.json
├── translated/               # PDF 翻译后内容
│   └── processing_tracker.json
├── toc_tree.json            # 目录结构（新版）
├── book_structure.json      # 目录结构（旧版）
├── refine_state.json        # Refine 阶段状态
├── translated_metadata.json  # 翻译后的元数据
└── batch_state.json         # Batch 任务状态（活动任务追踪）
```

**危险操作**：
- **永远不要用 `rm -rf output/{title}`** - 会删除所有翻译/处理进度
- title 配置直接影响目录路径，修改 title 会指向新目录
- **不要删除 `input_original.pdf`** - 这是原始 PDF 的唯一备份

### 2. Config.yaml 的 title 字段（最关键）

**重要**：每次处理新书前必须修改 `config.yaml` 中的 `title` 字段！

```yaml
title: "书名"  # 这个字段决定 output 目录名
```

**风险**：
- 如果 title 未修改，新书数据会覆盖或混入旧书目录
- 系统有冲突检测机制，但选择"覆盖"会导致数据丢失
- 用相同 title 处理不同 PDF 会覆盖 `input.pdf` 和 `input_original.pdf`

**最佳实践**：
- 处理新书前，先检查并修改 title
- 为不同书籍使用单独的配置文件：
  ```bash
  cp config.yaml config_book1.yaml
  # 修改 config_book1.yaml 中的 title
  uv run pdf2epub -c config_book1.yaml ocr-pages -i book1.pdf
  ```

### 3. 冲突处理选择

当系统提示目录冲突时：
- **选项 1（覆盖）**：危险！会删除现有数据
- **选项 2（创建新目录）**：推荐！安全选项
- **选项 3（中止）**：安全

### 4. Batch 任务状态管理

当使用 batch 模式（`mode: batch` 在 config.yaml 中配置）时：

**batch_state.json 文件**：
- 记录活动的 batch 任务状态
- 包含 `active_job_name`、`processing_keys` 等
- **中断后此文件会保留**，用于恢复

**中断处理**：
- Ctrl+C 会自动取消正在运行的 batch job
- 状态文件会被清理
- API 费用会在取消时停止

**重复提交保护**：
- 如果有活动的 batch job，不用 `--resume` 启动会报错
- 错误信息会提示使用 `--resume` 或手动取消

**手动取消 batch job**：
```bash
# 查看当前 batch jobs
uv run python -c "
from pdf2epub.utils.batch_utils import GeminiBatchClient
from pdf2epub.utils.common import load_config
config = load_config('config.yaml')
creds = config['credentials']['providers']['gemini-cf']
client = GeminiBatchClient(api_key=creds['api_key'], base_url=creds['base_url'], model='any')
for job in client.list_jobs(10):
    print(f'{job.name} - {job.state.value}')
"

# 取消特定 job
uv run python -c "
from pdf2epub.utils.batch_utils import GeminiBatchClient
# ... 同上 ...
client.cancel('batches/xxxxx')
"
```

### 5. --resume 参数（必须使用）

**所有命令在中断后必须使用 `--resume`**：

```bash
# 危险：中断后不用 --resume
uv run pdf2epub polish              # 处理到一半被中断
uv run pdf2epub polish              # 没有 --resume，从头开始覆盖！

# 正确：
uv run pdf2epub polish --resume     # 从上次中断处继续
```

支持 `--resume` 的命令：
- `ocr-pages --resume`
- `refine --resume`
- `polish --resume`
- `translate --resume`
- `translate-html --resume`

---

## 两种翻译工作流

### PDF 翻译（扫描版）

```bash
# 1. 页面级 OCR
uv run pdf2epub ocr-pages -i input.pdf [--resume]

# 2. 结构分析（生成 TOC）
uv run pdf2epub refine [--resume]

# 3. 内容润色
uv run pdf2epub polish [--resume]

# 4. 翻译（可选）
uv run pdf2epub translate --target-language Chinese [--resume]

# 5. 生成 EPUB
uv run pdf2epub build-epub [--translated]
```

**PDF 流程风险**：
- `ocr-pages --end-page N` 只处理部分页，但 `refine` 会基于不完整数据生成 TOC
- 如果用不同 PDF 重新运行 `ocr-pages`，会覆盖 `input.pdf` 和 `input_original.pdf`

### EPUB 翻译（文本版，保持原格式）

```bash
# 1. 翻译
uv run pdf2epub translate-html -i input.epub --target-language Chinese [--resume]

# 2. 生成翻译后 EPUB
uv run pdf2epub build-html-epub
```

**EPUB 流程风险**：
- `--limit N` 参数只翻译前 N 个文件（用于测试），其余文件保持原文
- 如果测试后直接 `build-html-epub`，会生成混合原文和翻译的 EPUB

---

## API Provider 配置

### 常用 Provider

| Provider | 类型 | 用途 |
|----------|------|------|
| `gemini` | google | 通过代理访问 Gemini（常规使用） |
| `vertex` | google | 官方 Vertex AI Express Mode |
| `deepseek` | openai | DeepSeek API |
| `anthropic` | anthropic | Claude API |
| `mistral` | mistral | Mistral OCR |
| `azure` | azure | Azure Document Intelligence |

### OCR Backend 配置

```yaml
ocr:
  backend: azure  # 可选：azure, vision, vertex, mistral, vllm

  # Vision API 需要 service account
  vision:
    service_account_key_path: sa-keys.json
```

**Backend 风险**：
- `vision` backend 需要 `sa-keys.json`，缺失会导致失败
- `azure` endpoint 末尾需要 `/`

### 翻译配置

```yaml
translation:
  source_language: French   # 源语言
  target_language: Chinese  # 目标语言
  models:
    - provider: gemini
      model: gemini-3-flash-preview
    - provider: deepseek      # 备用 provider
      model: deepseek-chat
```

---

## 检查命令

```bash
# 查看输出目录内容
ls -la "output/{title}/"

# 查看翻译进度（文件数量）
ls "output/{title}/translated_compressed/" | wc -l
ls "output/{title}/compressed_units/" | wc -l

# 查看进度跟踪文件
cat "output/{title}/polished_markdown/processing_tracker.json"

# 查看翻译后的元数据
cat "output/{title}/translated_metadata.json"

# 查看 OCR 进度
cat "output/{title}/pages/ocr_progress.json"
```

---

## 常见错误

| 错误 | 说明 | 处理 |
|------|------|------|
| "Object of type cython_function_or_method is not JSON serializable" | 警告 | 不影响翻译，忽略 |
| "503 UNAVAILABLE" | API 过载 | 系统会自动重试 |
| "Line count mismatch" | 验证失败 | 系统会自动重试 |
| "Resource not found in ZIP" | EPUB 资源缺失 | 警告，不影响翻译 |

---

## 禁止操作

1. **不要在不了解目录结构的情况下使用 `rm -rf`**
2. **不要在处理新书时忘记修改 title**
3. **不要在冲突提示时选择"覆盖"**（除非确定要删除旧数据）
4. **不要在中断后忘记使用 `--resume`**
5. **不要用 `--limit` 测试后直接 build EPUB**（先全量翻译）
6. **不要删除 `processing_tracker.json` 进度文件**
7. **不要提交 `config.yaml` 和 `sa-keys.json` 到 git**
8. **不要在 batch 任务运行时删除 `batch_state.json`**（会丢失任务追踪，导致重复提交或无法恢复）

---

## 安全备份建议

```bash
# 处理前备份
cp -r "output/{title}" "output/{title}.backup"

# 处理完成后备份
cp -r "output/{title}" "output/{title}.final"
```

---

## 风险等级总结

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| PDF 被不同 PDF 覆盖 | 高 | 为每本书用不同 title |
| Markdown 被重新处理覆盖 | 高 | 总是使用 --resume |
| API 凭证泄露 | 高 | 加入 .gitignore |
| 处理不完整导致结构错误 | 中 | 完整运行所有阶段 |
| 旧目录遗留导致混乱 | 中 | 修改 title 后清理旧目录 |
| 进度文件损坏 | 中 | 备份 output/ 目录 |
