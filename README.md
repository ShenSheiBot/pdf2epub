
# pdf2epub

将外语的学术学籍或者纵排日语书籍的扫描件转换成epub格式，保留完备的目录、注音、脚注、插图、表格（公式待支持）等信息，具备完备的链接跳转功能，使其尽可能接近出版社原epub的排版。

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
2. **markdown ocr**：对于非纵排日语，使用mistral-OCR将每页转换成带有插图的markdown。
3. **markdown ocr (jp)**: 对于纵排日语，使用多模态LLM将每页转换成markdown，使用cloudvision的版面分析工具解出插图。
4. **polish**: 使用LLM建立正确的链接跳转，消除OCR错误、多余的页眉页脚、页间分隔符、空白页，整理跨页的标题等级等。
5. **epub**: 将markdown和图片打包为epub格式。


## 推荐LLM：

无审核压力：gemini-2.5-pro

有审核压力：claude-sonnet-4-20250514

## 安装

### 依赖要求
- Python 3.11+
- Poetry (包管理器)
- Google Cloud Vision API 账户（用于日语纵排书籍的插图检测）

### 安装步骤

1. 克隆仓库

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

5. （仅日语纵排书籍）配置 Google Cloud Vision
   - 在 [Google Cloud Console](https://console.cloud.google.com/apis/credentials) 创建服务账户
   - 下载 JSON 密钥文件
   - 保存为项目根目录的 `sa-keys.json`

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

# OCR 模型配置（按顺序尝试）
ocr_models:
  - provider: gemini
    model: gemini-2.5-pro
    max_retries: 1
  - provider: anthropic
    model: claude-sonnet-4-20250514
    max_retries: 2
```

### 2. 基本工作流程

#### 步骤 1: 分析 PDF 结构
```bash
poetry run python src/breakdown.py -i input.pdf
```
生成 `output/{book_title}/book_structure.json`

#### 步骤 2: OCR 转换

**对于横排文本（学术书籍等）：**
```bash
poetry run python src/ocr_chapters.py
```

**对于纵排日语书籍：**
```bash
poetry run python src/ocr_chapters_jp.py
```

参数说明：
- `--resume`: 从上次中断处继续
- `--max-workers N`: 设置并行处理线程数（默认4）

#### 步骤 3: 内容润色
```bash
poetry run python src/polish_ocr_markdown.py
```
- 自动建立链接跳转
- 修正 OCR 错误
- 去除页眉页脚
- 整理章节标题

#### 步骤 4: 生成 EPUB
```bash
poetry run python src/generate_epub.py
```
最终 EPUB 文件保存在 `output/{book_title}/output.epub`

### 3. 高级配置

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
```

### 4. 故障排除

#### OCR 失败
- 检查 API 配额
- 降低 `max_workers` 减少并发
- 使用 `--resume` 从失败处继续

#### 审核问题
- 配置多个模型提供商
- Gemini 被阻止时会自动切换到 Anthropic

#### 内存不足
- 减少 `max_workers`
- 降低 `zoom_factor`

### 5. 输出结构
```
output/
└── {book_title}/
    ├── book_structure.json    # 书籍结构
    ├── ocr_markdown/          # OCR 原始结果
    │   ├── chapter_1.md
    │   └── ...
    ├── polished_markdown/     # 润色后内容
    │   ├── chapter_1.md
    │   └── ...
    ├── images/                # 提取的插图
    └── output.epub           # 最终 EPUB