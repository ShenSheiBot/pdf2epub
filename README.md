# pdf2epub

将外语的学术书籍或者纵排日语书籍的扫描件转换成epub格式，保留完备的目录、注音、脚注、插图、表格（公式待支持）等信息，具备完备的链接跳转功能，使其尽可能接近出版社原epub的排版。

## 技术优势
经测试，本项目效果远强于各大商业软件的直接转换效果，同时因为其基于OCR的特性，不会因出版社更新DRM机制而失效。

### Demo
转换前：http://biopolitics.kom.uni.st/Michel%20Foucault/The%20Foucault%20Reader%20(149)/The%20Foucault%20Reader%20-%20Michel%20Foucault.pdf

转换后：https://raw.githubusercontent.com/ShenSheiBot/pdf2epub/refs/heads/main/example.epub

## 局限性
因为其复杂性和对多模态LLM的依赖，转换速度较慢并有可能会因为LLM的审核原因失败。建议使用多家不同LLM提供商混合使用。

对于纵排日语，需要扫描文件的质量较高且为*白底*。

因为逐页进行转换，要求书的新章节*新起一页*，否则章节的最后部分可能会被顺延到下一章节。如果扫描件是每两页一扫描的pdf，建议拆分成单页pdf再操作。

## 思路：
1. **breakdown**：使用具有极大上下文的 gemini 解出全书基于pdf页数的目录结构
2. **markdown ocr**：对于非纵排日语，使用多模态LLM将每页转换成带有插图的markdown。
3. **markdown ocr (jp)**: 对于纵排日语，使用专门的OCR后端处理，支持振假名(furigana)识别和插图提取。
4. **polish**: 使用LLM建立正确的链接跳转，消除OCR错误、多余的页眉页脚、页间分隔符、空白页，整理跨页的标题等级等。
5. **epub**: 将markdown和图片打包为epub格式。


## 推荐LLM：

无审核压力：gemini-2.5-pro

有审核压力：claude-sonnet-4-20250514，或者从gemini回落到claude

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
- Poetry (包管理器)
- 至少一个OCR后端的API账户

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/pdf2epub.git
cd pdf2epub
```

2. 安装 Poetry（如果未安装）
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装项目依赖
```bash
poetry install
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
jp_ocr_backend: azure  # 使用Azure后端

# Azure配置
azure_endpoint: https://your-resource.cognitiveservices.azure.com/
azure_api_key: your-azure-api-key
```

#### Google Cloud Vision
```yaml
# config.yaml
jp_ocr_backend: vision  # 使用Vision后端

# Google Cloud配置
service_account_key_path: /path/to/sa-keys.json
# 或设置环境变量 GOOGLE_APPLICATION_CREDENTIALS
```

#### Vision Language Models (VLLM)
```yaml
# config.yaml
jp_ocr_backend: vllm  # 使用VLLM后端

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

# API 密钥
google_api_key: your-google-api-key
anthropic_api_key: your-anthropic-api-key  # 可选，作为备选

# 选择日语OCR后端 (azure/vision/vllm)
jp_ocr_backend: vision

# OCR 模型配置（用于非日语书籍）
ocr_models:
  - provider: gemini
    model: gemini-2.5-pro
    max_retries: 1
  - provider: anthropic
    model: claude-3-5-haiku-20241022
    max_retries: 2
```

### 2. 基本工作流程（统一CLI）

所有功能通过统一的CLI入口访问：

#### 步骤 1: 分析 PDF 结构
```bash
python -m pdf2epub.cli breakdown -i input.pdf
```
生成 `output/{book_title}/book_structure.json`

#### 步骤 2: OCR 转换

**对于普通书籍：**
```bash
python -m pdf2epub.cli ocr
```

**对于日语纵排书籍：**
```bash
python -m pdf2epub.cli ocr --japanese --backend vision
```

参数说明：
- `--backend [azure|vision|vllm]`: 覆盖配置文件中的后端选择
- `--resume`: 从上次中断处继续
- `--max-workers N`: 设置并行处理线程数（默认4）

#### 步骤 3: 内容润色
```bash
python -m pdf2epub.cli polish
```

针对不同内容类型：
```bash
# 学术书籍（带脚注和引用）
python -m pdf2epub.cli polish --content-type academic

# 日语书籍（保留振假名）
python -m pdf2epub.cli polish --content-type japanese

# 自动检测内容类型
python -m pdf2epub.cli polish --content-type auto
```

#### 步骤 4: 生成 EPUB
```bash
python -m pdf2epub.cli epub
```
最终 EPUB 文件保存在 `output/{book_title}/output.epub`

### 3. 翻译功能

#### 实体提取（可选，用于保持翻译一致性）

对于包含大量专有名词的书籍（如日语轻小说），可以先提取实体：

```bash
# 提取人物、地点、术语等实体
python -m pdf2epub.cli extract-entities -i input.pdf --source-lang Japanese --target-lang Chinese
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
python -m pdf2epub.cli translate --target-language Chinese

# 强制使用实体参考（即使文件不存在也会报错）
python -m pdf2epub.cli translate --target-language Chinese --use-entities

# 强制不使用实体（即使文件存在）
python -m pdf2epub.cli translate --target-language Chinese --no-entities

# 指定源语言和目标语言
python -m pdf2epub.cli translate --source-language Japanese --target-language Chinese
```

**注意**：如果 `translation_entities.json` 文件存在，翻译器会自动使用它以保持一致性。

### 4. 完整工作流程示例

#### 日语轻小说翻译流程
```bash
# 1. 分析结构
python -m pdf2epub.cli breakdown -i manga.pdf

# 2. 提取翻译实体
python -m pdf2epub.cli extract-entities -i manga.pdf

# 3. 日语OCR
python -m pdf2epub.cli ocr --japanese --backend vision

# 4. 日语内容润色
python -m pdf2epub.cli polish --content-type japanese

# 5. 翻译成中文（自动使用已提取的实体）
python -m pdf2epub.cli translate

# 6. 生成EPUB
python -m pdf2epub.cli epub
```

#### 学术书籍翻译流程
```bash
# 1. 分析结构（添加页码标记）
python -m pdf2epub.cli breakdown -i thesis.pdf --add-page-numbers

# 2. OCR提取
python -m pdf2epub.cli ocr

# 3. 学术内容润色（保留脚注）
python -m pdf2epub.cli polish --content-type academic

# 4. 翻译
python -m pdf2epub.cli translate --target-language Chinese

# 5. 生成EPUB
python -m pdf2epub.cli epub
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
    ├── book_structure.json    # 书籍结构
    ├── input_original.pdf     # 原始PDF副本
    ├── ocr_markdown/          # OCR 原始结果
    │   ├── chapter_1.md
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
    ├── progress.json          # OCR进度跟踪
    ├── polish_progress.json  # 润色进度跟踪
    ├── translation_progress.json  # 翻译进度跟踪
    └── output.epub           # 最终 EPUB
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可

MIT License