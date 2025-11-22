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

### 推荐工作流 (Recommended - uses toc_tree.json)

1. **ocr-pages**：逐页进行 OCR，使用多模态 LLM 将每页转换成带有插图的 markdown
2. **refine**：智能分析 TOC 结构，验证章节边界，生成精确的 toc_tree.json（支持无限层级嵌套）
3. **polish**：使用 LLM 建立正确的链接跳转，消除 OCR 错误、页眉页脚等
4. **translate**：（可选）使用 LLM 翻译成目标语言
5. **build-epub**：基于 toc_tree.json 生成 EPUB

### 旧版工作流 (Legacy - uses book_structure.json) ⚠️ DEPRECATED

1. ~~**breakdown**：使用 Gemini 分析 PDF 结构~~
2. ~~**ocr**：章节级 OCR~~
3. **polish**：润色
4. ~~**epub**：生成 EPUB~~


## 推荐LLM：

- breakdown / entity extraction：仅支持 gemini-2.5-pro，一般不会有审核问题
- polish：deepseek-chat 或 claude-sonnet-4，仅当无审核压力时推荐gemini-2.5-pro
- translate：deepseek-chat，claude-sonnet-4 翻译流畅度较差，仅当无审核压力时推荐gemini-2.5-pro

## 日语OCR架构

### OCR后端支持

本项目支持三种OCR后端用于日语纵排文本识别：

#### 1. **Azure Document Intelligence** (`azure`)
- 效果最佳，支持振假名(furigana)检测和重组，基本保证准确
- 需要Azure账户和API密钥

#### 2. **Google Cloud Vision** (`vision`)
- 效果次佳，正文偶有漏字，振假名偶有错漏
- 需要Google Cloud账户和服务账户密钥

#### 3. **Vision Language Models** (`vllm`)
- Gemini 识别效果较佳，但经常“自由发挥”，添加不存在的振假名，且审核严格，不推荐
- Anthropic 识别效果较差，虽然审核宽松，更不推荐
- VLLM 整体识别速度缓慢且费用较高，胜在输出文本连贯，但仍不能完全摆脱后处理需求，故整体仅作为备用方案

## 安装

### 依赖要求
- Python 3.11+
- UV (包管理器)
- 至少一个OCR后端的API账户

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

### 配置OCR后端

#### Azure Document Intelligence
```yaml
# config.yaml
ocr_backend: azure  # 使用Azure后端

# Azure配置
azure_endpoint: https://your-resource.cognitiveservices.azure.com/
azure_api_key: your-azure-api-key
```

#### Google Cloud Vision
```yaml
# config.yaml
ocr_backend: vision  # 使用Vision后端

# Google Cloud配置
service_account_key_path: /path/to/sa-keys.json
# 或设置环境变量 GOOGLE_APPLICATION_CREDENTIALS
```

#### Vision Language Models (VLLM)
```yaml
# config.yaml
ocr_backend: vllm  # 使用VLLM后端

# VLLM模型配置
ocr_vllm_models:
  - provider: anthropic
    model: claude-sonnet-4-20250514
    max_retries: 2
  # 或使用Gemini
  - provider: gemini
    model: gemini-2.5-pro
    max_retries: 1

# API密钥
anthropic_api_key: your-anthropic-key
google_api_key: your-google-key
```

## 使用方法

### 1. 配置文件设置

首先在 `config.yaml` 里配置以下信息：

```yaml
# 书籍信息
title: 书名
author: 作者名

# LLM API 密钥
google_api_key: your-google-api-key
anthropic_api_key: your-anthropic-api-key  # 可选
anthropic_base_url: https://api.anthropic.com  # 可选，自定义端点
openai_api_key: your-openai-api-key  # 可选，支持兼容API如DeepSeek
openai_base_url: https://api.deepseek.com/v1  # 可选，自定义端点
openai_model: deepseek-chat  # 可选，模型名称

# 选择OCR后端 (mistral/vertex/vllm/azure/vision)
ocr_backend: vision  # 日语推荐azure或vision

# 高级配置
num_retries: 1  # API重试次数
max_backoff_seconds: 30  # 最大退避时间
max_concurrent_workers: 8  # 最大并发API调用数

# 实体提取模型（可选）
entity_extraction_model: gemini-2.5-pro

# 翻译配置
source_language: Japanese
target_language: Chinese

# 处理模型配置
polish_models:
  - provider: anthropic
    model: claude-sonnet-4-20250514
    max_retries: 2

translation_models:
  - provider: openai
    model: deepseek-chat
    max_retries: 2
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

#### 多模型自动切换
系统支持在模型失败或触发安全审核时自动切换：

```yaml
polish_models:
  - provider: gemini
    model: gemini-2.5-pro
    max_retries: 1  # 瞬时错误重试次数
  - provider: anthropic
    model: claude-sonnet-4-20250514
    max_retries: 2
```

#### OCR 优化设置
```yaml
ocr_settings:
  zoom_factor: 1.5  # 提高图像质量（1.0-3.0）
  max_workers: 8    # 增加并行数
  illustration_padding: 30  # 插图检测边距
  illustration_min_black_pixels: 200  # 插图最小像素数
```

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
    ├── toc_tree.json          # TOC结构（推荐工作流，支持无限层级）
    ├── book_structure.json    # 书籍结构（旧版工作流）
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

## ⚠️ 已弃用命令 (Deprecated Commands)

以下命令属于旧版工作流，仍可运行但不推荐使用：

| 旧命令 | 替代命令 | 说明 |
|--------|----------|------|
| `breakdown` | `ocr-pages` + `refine` | 旧版使用 Gemini 分析结构，新版使用边界验证 |
| `ocr` | `ocr-pages` + `refine` | 旧版章节级 OCR，新版页级 OCR + 智能拆分 |
| `epub` | `build-epub` | 旧版使用 book_structure.json，新版使用 toc_tree.json |

### 旧版工作流（不推荐）
```bash
# ⚠️ DEPRECATED - 以下命令仍可使用但不推荐
uv run pdf2epub breakdown -i input.pdf  # 会显示弃用警告
uv run pdf2epub ocr                      # 会显示弃用警告
uv run pdf2epub polish
uv run pdf2epub epub                     # 会显示弃用警告
```

### 迁移到新工作流
如果你之前使用旧工作流，建议迁移到新工作流以获得：
- 更精确的章节边界检测
- 支持无限层级的 TOC 嵌套
- 更好的错误恢复和进度跟踪

## 贡献

欢迎提交Issue和Pull Request！
关注[甚谁](https://www.zhihu.com/people/sakuraayane_justice)谢谢喵！

## 许可

MIT License