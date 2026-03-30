# Agent Runner 设计文档

## 核心洞察

Agent 就是一个拿着工具干活的 LLM。它不需要我们用 Python 硬编码验证逻辑、缝合策略、游标模型——给它 bash 和文件工具，它自己会看文件、判断问题、操作接缝。

三个场景（JSON 目录、HTML 翻译、章节翻译）是**同一个 agent，不同的 prompt**。工具相同、沙盒相同、循环相同、decision 类型相同。不需要 Protocol、不需要三个类、不需要各自的 validate_and_repair。

---

## Agent 的工具

六个标准工具 + 一个 structured result：

| 工具 | 说明 |
|------|------|
| **bash** | 在沙盒工作目录中执行任意命令。主力工具。stdout 超过 32KB 截断并提示用 read 查看完整内容 |
| **read** | read(path, offset, limit) — 精确读取文件区域 |
| **edit** | edit(path, old_string, new_string, replace_all) — 精确文本替换，replace_all 可全局替换（如单引号换双引号） |
| **write** | write(path, content) — 覆写文件 |
| **glob** | glob(pattern) — 按 pattern 找文件 |
| **grep** | grep(pattern, path) — 搜索文件内容 |

Result type（不是工具，是 agent 的最终输出）：

```python
class Decision(BaseModel):
    action: Literal["continue", "complete"]
    file_path: str
```

工具完整保留标准形态，不阉割、不魔改。Agent 在这些工具上训练过，改了反而降低能力。

---

## 沙盒

### Phase 1: 安全临时目录

使用 `tempfile.mkdtemp()` 创建隔离工作目录：

- 所有工具的路径操作限制在工作目录内
- bash 命令通过 `subprocess.run(cwd=work_dir)` 执行
- originals/ 保护通过沙盒的 `allowWrite` 策略实现（不是 chmod）
- bash 命令设置超时（默认 30 秒）
- stdout 截断到 32KB 防止 token 爆炸

### Phase 2: srt CLI 增强（可选）

通过 `@anthropic-ai/sandbox-runtime` CLI 包装 bash 命令：

```json
{
  "network": {"allowedDomains": []},
  "filesystem": {
    "allowWrite": ["/tmp/agent_work_xxx/workspace"],
    "denyRead": ["~/.ssh", "~/.aws", "~/.config"]
  }
}
```

- macOS 用 sandbox-exec，Linux 用 bubblewrap
- 网络完全禁止
- 读取禁止敏感目录
- 写入仅允许 workspace/

沙盒层是可插拔的抽象——Phase 1 用 subprocess，Phase 2 可无缝切换到 srt。

---

## Agent Loop

```
调用方                        Agent Runner                         Agent (LLM)
  │                               │                                    │
  │  run_agent_loop(prompt, ...)  │                                    │
  │──────────────────────────────>│                                    │
  │                               │  1. 创建临时工作目录                │
  │                               │  2. generate_fn() → raw_output.txt │
  │                               │  3. 调用 pydantic-ai agent         │
  │                               │──────────────────────────────────>│
  │                               │     agent 用 bash/read/edit 操作   │
  │                               │     工作目录里的文件               │
  │                               │<──────────────────────────────────│
  │                               │     Decision(continue, file_path)  │
  │                               │                                    │
  │                               │  4. 读 file_path 作为 prefix       │
  │                               │  5. generate_fn(prefix=...) →      │
  │                               │     continuation_001.txt           │
  │                               │  6. 再次调用 agent（全新 run）      │
  │                               │──────────────────────────────────>│
  │                               │     agent 检查接缝、操作文件       │
  │                               │<──────────────────────────────────│
  │                               │     Decision(complete, file_path)  │
  │                               │                                    │
  │                               │  7. 读 file_path 作为最终结果      │
  │                               │  8. 销毁临时目录                   │
  │<──────────────────────────────│                                    │
  │  返回最终内容                  │                                    │
```

### 关键设计决策

1. **每轮 agent 是全新的 `agent.run()`**，不保留上一轮对话历史。Agent 只通过 workspace 文件感知状态。这样更简单、更便宜（不需要维护长对话），也避免了 context window 膨胀。

2. **`run_agent_loop` 替换外层重试循环**，不嵌套在里面。对于 `adaptive_pdf_call.py`，agent loop 替换的是 lines 533-551 的 JSON retry 循环。外层的 validation loop（语义验证）保留。

3. **`generate_fn` 的合约**：`generate_fn(prefix: Optional[str] = None) -> str`。调用方负责将 prefix 构造为多轮消息（user msg 不变 + bot msg = prefix + user msg = "继续"）。Runner 不需要了解消息构造细节。

4. **必须关闭 JSON mode**：使用 agent loop 时，generate_fn 不能设置 `response_mime_type = "application/json"`，因为 continuation 片段不是有效 JSON。JSON 语法验证由 agent 自行完成。

### 每轮 agent 看到的工作目录

首轮：
```
/work/
  originals/                # 只读区（沙盒策略保护，不可写入）
    raw_output.txt          # generate_fn 的原始输出
  workspace/                # 可写区，agent 自由操作
```

第 N 轮（continuation 后）：
```
/work/
  originals/                # 只读（沙盒策略保护）
    raw_output.txt          # 最初的原始输出
    continuation_001.txt    # 第 1 次续写的原始输出
    continuation_002.txt    # 第 2 次续写的原始输出
  workspace/                # 可写
    ...                     # agent 自己创建的任何文件
```

**originals/ 的只读保护**通过沙盒的 `allowWrite` 策略实现——只有 workspace/ 目录在写入白名单中。比 chmod 444 更可靠，因为 bash 可以 `chmod u+w` 绕过文件权限，但无法绕过沙盒策略。Agent 要操作内容必须先 copy 到 workspace/。

### Continuation 的 message 结构

为最大化 Gemini implicit cache（90% 折扣），continuation 的 LLM 调用结构：

```
Message 1 [User]:  原始 prompt + PDF/内容（不变，命中 cache）
Message 2 [Bot]:   agent 清理后的 prefix（前缀不变，命中 cache）
Message 3 [User]:  "从这里继续"
```

Gemini implicit cache 是 token 级前缀匹配：只要前缀相同就命中，改了中间则从改动点之后全部 miss。所以：
- Message 1 始终不变 → 100% cache hit
- Message 2（bot prefix）如果 agent 只截尾不改中间 → 前半部分也能 cache hit
- Agent prompt 中建议"能只截尾就别改中间"，但不强制——agent 觉得中间有问题就改

### Agent 模型选择

Agent 使用廉价模型（Gemini 2.5 Flash / Haiku）：
- Agent 的任务是检查和修复，不是生成内容
- 需要工具使用能力和指令遵循能力，不需要创造力
- 配置方式：从 config.yaml 的 `refine.verification.provider` + `model` 读取

### 请求数限制

Agent 每轮的工具调用受 `refine.agent_request_limit` 限制（默认 100），防止 agent 陷入无限循环。

---

## Prompt 模板

三个场景，三个 system prompt。工具和循环完全相同。

### JSON 目录（refine）

```
你是一个 JSON 验证和修复 agent。

工作目录结构：
- originals/raw_output.txt — LLM 生成的原始输出（只读）
- originals/continuation_NNN.txt — 续写的原始输出（只读，如果有的话）
- workspace/ — 你的工作区（可写）

你的任务：确保最终产出一个语法正确、结构完整的 JSON 文件。

流程：
1. 先把需要的文件从 originals/ 复制到 workspace/
2. 用 python3 -c "import json; ..." 检查 JSON 语法
3. 如果语法有问题但内容完整（如单引号、尾部逗号等），用工具修复语法
4. 如果截断了（JSON 不闭合），找到最后一个完整的 chapter 对象，
   截掉后面的残片，然后 continue
5. 如果语法正确且结构完整，complete

续写处理（当 originals/ 中出现 continuation_NNN.txt）：
- 续写文件开头可能有垃圾文本（如"接上文"），需要清理
- 续写可能跟前文有重叠（重复的 chapter），需要按 title/start_page 去重
- 用 cat 拼接、edit 清理接缝

幻觉检测：
- 检查页码是否连续递增（突然跳了几百页 → 可能是幻觉）
- 检查 title 风格是否一致（突然从中文变英文、格式突变 → 可能是幻觉）
- 如果发现幻觉，回退到最后一个可信的 chapter，截掉幻觉部分，然后 continue

格式一致性（续写拼接后）：
- 检查前后 chapter 的字段是否一致（有没有续写部分突然多了/少了字段）
- 检查 level、children 等结构风格是否统一
- 如果不一致，用 edit 统一格式

注意：
- 修改前缀时尽量只截尾、不改中间，有利于续写时的 Gemini cache 命中
  （但如果中间确实有问题，该改就改）
- originals/ 是只读的，永远不要直接修改
```

### 压缩 HTML 翻译

```
你是一个 HTML 翻译验证和修复 agent。

工作目录中有 LLM 翻译的压缩 HTML 文件。原文是 <div> 包裹的行，每行一个翻译单元。你的任务：
1. 检查翻译输出的 <div> 行数是否与原文一致
2. 如果截断了（行数不足），保留已翻译的行，截掉不完整的最后一行
3. 如果行数匹配，直接 complete

（与 JSON 类似的续写和接缝处理指令）
```

### 章节翻译

```
你是一个翻译验证和修复 agent。

工作目录中有 LLM 翻译的章节内容。你的任务：
1. 检查翻译是否看起来完整（末尾是否自然结束）
2. 如果截断了，找到最后一个完整的段落
3. 如果完整，直接 complete

（与 JSON 类似的续写和接缝处理指令）
```

---

## 实现结构

```
pdf2epub/core/whole/
  __init__.py
  runner.py          # run_agent_loop() — 通用循环
  sandbox.py         # 沙盒抽象层（Phase 1: subprocess, Phase 2: srt CLI）
  tools.py           # bash, read, edit, write, glob, grep 的 pydantic-ai tool 定义
  prompts/
    __init__.py
    json_refine.py   # JSON 目录的 system prompt
    html_translate.py # HTML 翻译的 system prompt（Phase 2）
    chapter_translate.py # 章节翻译的 system prompt（Phase 2）
```

### runner.py 的核心接口

```python
async def run_agent_loop(
    generate_fn: Callable[..., str],
    system_prompt: str,
    agent_model: Model,            # pydantic-ai Model 实例
    max_continuations: int = 5,
    request_limit: int = 100,
    work_dir: Optional[Path] = None,
) -> str:
    """
    通用 agent 辅助生成循环。

    1. 调用 generate_fn() 获取初始输出
    2. 在工作目录中创建 originals/raw_output.txt
    3. 启动 pydantic-ai agent（带 system_prompt 和标准工具）
    4. Agent 返回 Decision：
       - complete(file_path) → 读取文件，返回内容
       - continue(file_path) → 读取文件作为 prefix，
         调用 generate_fn(prefix=...) 获取续写，
         存为 continuation_NNN.txt，再次运行 agent（全新 run）
    5. 超过 max_continuations 次仍未 complete → 抛异常

    Args:
        generate_fn: 生成函数。签名 generate_fn(prefix=None) -> str。
                     调用方负责将 prefix 构造为适当的多轮消息。
        system_prompt: Agent 的 system prompt
        agent_model: pydantic-ai Model 实例（如 GoogleModel）
        max_continuations: 最大续写次数
        request_limit: Agent 每轮最大工具调用次数
        work_dir: 工作目录（默认自动创建临时目录）

    Returns:
        最终内容（agent complete 的文件内容）

    Raises:
        AgentLoopExhausted: 超过 max_continuations 次仍未完成
    """
```

### 调用方集成

```python
# adaptive_pdf_call.py — JSON 目录截断修复
# 替换原有的 json_retry 循环（lines 533-551）
# 注意：必须去掉 config.response_mime_type = "application/json"

def _build_generate_fn(self, parts, config, op_name):
    """构造 generate_fn，闭包捕获 parts/config/op_name"""
    def generate_fn(prefix=None):
        if prefix is None:
            contents = parts
        else:
            # 多轮消息：user msg 不变 + bot prefix + "继续"
            contents = build_continuation_contents(parts, prefix)
        return self.client.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
            operation_name=op_name,
        )
    return generate_fn

result = await run_agent_loop(
    generate_fn=self._build_generate_fn(parts, config, op_name),
    system_prompt=JSON_REFINE_PROMPT,
    agent_model=self._get_agent_model(),
)
parsed = json.loads(result)
```

### Agent 模型获取

```python
def _get_agent_model(self):
    """从 config.yaml 获取 agent 使用的模型"""
    config = load_config()
    verification = config.get('refine', {}).get('verification', {})
    provider_name = verification.get('provider', 'poe')
    model_name = verification.get('model', 'Gemini-2.5-Flash')
    # 构造 pydantic-ai Model 实例
    ...
```

---

## 与 whole-mode-design.md 的关系

whole-mode-design.md 描述的是宏观架构：两个正交维度、strategy 是外层 transport 是内层、`_process_whole` 与 `_process_single` 的关系、per-unit strategy、whole→split fallback。

本文档描述的是 agent 的具体实现：工具、沙盒、循环、prompt。是 whole-mode-design.md 中 "run_agent_loop" 的展开。

两个文档互补：宏观架构不变，agent 的实现细节从 "Protocol + 三个类" 简化为 "一个 runner + 三个 prompt"。

---

## Phase 1 实现范围

Phase 1 只实现 JSON 目录场景：

1. **runner.py** — `run_agent_loop()` 通用循环
2. **sandbox.py** — subprocess 沙盒（tmpdir + cwd + 超时 + 输出截断）
3. **tools.py** — 六个标准工具的 pydantic-ai tool 定义
4. **prompts/json_refine.py** — JSON 目录的 system prompt
5. **adaptive_pdf_call.py 集成** — 替换 JSON retry 循环

不实现：
- HTML 翻译 prompt（Phase 2）
- 章节翻译 prompt（Phase 2）
- srt CLI 沙盒增强（Phase 2）
- `_process_whole` executor 集成（Phase 3）
