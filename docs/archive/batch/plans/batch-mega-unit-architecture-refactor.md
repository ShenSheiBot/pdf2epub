# Batch Mega Unit 架构重构

## 核心洞察

当前架构把 Batch 作为一个**全局特殊状态机**，与 Online 形成二元对立。这导致了：
- `_process_batch` 有自己的 for loop（~400 行）
- Batch 验证失败不 requeue，直接标记 `validation_failed`
- `execute()` 里有 ~130 行的 fallback 胶水代码
- Batch quota 管理与 Online 完全分裂
- 全局单一的 `batch_state.json`，复杂的 resume 逻辑

**新洞察**：Batch Job 只是一个"大一点的 Unit"（**Mega Unit**）。它应该：
- 作为一个 Future 提交到 ThreadPoolExecutor（占用一个 worker slot）
- 有自己的状态文件（不是全局的 batch_state.json）
- ID 由内部 unit IDs 唯一确定（相同 pending = 相同 ID = 自然 Resume）
- 返回结果后，每个内部 unit 通过统一的 `_handle_result()` 处理
- 失败的 unit 自然 requeue，下一轮根据 chain[0] 决定走 batch 还是 online

## 为什么 Batch 和 Online 可能同时存在

一个典型场景：200 个 units 提交 batch job

```
第一轮：chain = [batch:gemini, online:gemini, online:anthropic]
  提交 200 个到 batch job
  - 180 成功 → promote
  - 10 个 safety error → apply_effect(remove_provider=gemini)
                       → chain 变成 [online:anthropic]
  - 10 个 truncation → apply_effect 只减 quota
                     → chain 仍是 [batch:gemini, online:gemini, ...]

第二轮：
  - 10 个 safety blocked：chain[0] = online:anthropic → 走 online
  - 10 个 truncation：chain[0] = batch:gemini → 继续 batch（如果 >= 5）
```

所以第二轮确实可能**同时有 batch 和 online**，每个 unit 的 chain 在 `apply_effect` 后变得不同。

## Mega Unit 设计

### ID 计算

```python
def _get_mega_unit_id(self, unit_ids: List[str]) -> str:
    """相同的 pending units = 相同的 ID = 自然 Resume"""
    sorted_ids = sorted(unit_ids)
    return "batch:" + hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()[:16]
```

### 状态文件结构

每个 mega unit 有自己的状态文件，就像每个普通 unit 有自己的输出文件：

```
output/{title}/
├── raw/
│   ├── chapter_1.md              # 普通 unit 输出
│   └── ...
├── batch_states/
│   ├── batch_a3f2b1c4.json       # mega unit 状态
│   └── batch_e5d6f7g8.json       # 另一个 mega unit（如果存在）
```

每个 `batch_{id}.json` 只需要：
```json
{
  "job_name": "batches/xxx",
  "job_state": "RUNNING"
}
```

### 生命周期（与普通 Unit 完全对称）

| 普通 Unit | Mega Unit |
|-----------|-----------|
| ID = `"chapter_1"` | ID = `"batch:a3f2b1c4"` |
| 输出文件 = `raw/chapter_1.md` | 状态文件 = `batch_states/batch_a3f2b1c4.json` |
| 检查 tracker 是否 completed | 检查状态文件是否存在 + job 完成 |
| 已完成 → 跳过 | 已完成 → 直接返回缓存结果 |
| 运行中 → 等待 Future | 运行中 → 等待 job 完成 |
| 无 resume → 重新处理 | 无 resume → cancel 所有旧 job |

### Resume 流程示例

```
第一轮：100 units → mega_id = "batch:abc123"
  提交 job
  保存 batch_states/batch_abc123.json

中断（Ctrl+C 或进程崩溃）

Resume：
  Tracker 检查：0 个 completed
  收集：100 个 pending
  计算：mega_id = "batch:abc123"（相同！）
  检查 batch_abc123.json：有活跃 job
  → 继续等待该 job
  → 处理结果
  → 80 成功写入 raw/，更新 tracker
  → 20 失败 requeue
  → 清除 batch_abc123.json

下一轮：
  Tracker 检查：80 个 completed
  收集：20 个 pending
  计算：mega_id = "batch:def456"（不同）
  检查 batch_def456.json：不存在
  → 提交新 job，保存 batch_def456.json
```

## 主循环重构

```python
def execute(self, units, context_base, resume_batch=False):
    # 初始化（不变）
    unit_states = {u.id: create_unit_state(...) for u in units}
    pending = {u.id for u in units}
    futures: Dict[Future, str] = {}           # online futures
    batch_futures: Dict[Future, List[str]] = {}  # mega unit futures

    # 不 resume 时，cancel 所有旧 batch job
    if not resume_batch:
        self._cancel_all_batch_jobs()

    with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
        while pending or futures or batch_futures:
            # 1. 收集 ready units
            ready_ids = self._get_ready_ids(pending, completed, in_progress, unit_states)

            # 2. 分流：根据 chain[0].mode
            batch_queue = []
            for uid in ready_ids:
                pending.discard(uid)
                state = unit_states[uid]

                if state.is_aggregation:
                    # 聚合单元直接处理（不变）
                    self._handle_aggregation(uid, state, ...)
                    continue

                if state.get_current_mode() == "batch":
                    batch_queue.append(uid)
                else:
                    # Online：提交单个任务
                    in_progress.add(uid)
                    future = pool.submit(self._process_single, ...)
                    futures[future] = uid

            # 3. 处理 batch queue
            if batch_queue:
                if len(batch_queue) >= self._batch_threshold:
                    # 够数量：作为 mega unit 提交
                    future = pool.submit(
                        self._process_batch_as_unit,
                        batch_queue, resume_batch, ...
                    )
                    batch_futures[future] = batch_queue
                    in_progress.update(batch_queue)
                else:
                    # 不够：强制移除 batch entries，转 online
                    for uid in batch_queue:
                        unit_states[uid].remove_batch_entries()
                        in_progress.add(uid)
                        future = pool.submit(self._process_single, ...)
                        futures[future] = uid

            # 4. 检查终止条件
            all_futures = set(futures.keys()) | set(batch_futures.keys())
            if not all_futures:
                if pending:
                    # 死锁检测
                    failed.update(pending)
                    pending.clear()
                break

            # 5. 等待任意完成
            done, _ = wait(all_futures, return_when=FIRST_COMPLETED, timeout=1.0)

            # 6. 处理完成的 futures
            for future in done:
                if future in batch_futures:
                    # Mega unit 完成：返回多个结果
                    batch_unit_ids = batch_futures.pop(future)
                    in_progress.difference_update(batch_unit_ids)

                    for uid, result in future.result():
                        self._handle_result(uid, result, unit_states[uid], ...)
                        # 失败会自动 requeue 到 pending
                else:
                    # 普通 online unit 完成
                    uid = futures.pop(future)
                    in_progress.discard(uid)
                    result = future.result()
                    self._handle_result(uid, result, unit_states[uid], ...)
```

## Mega Unit 实现

```python
def _process_batch_as_unit(
    self,
    unit_ids: List[str],
    resume_batch: bool,
    unit_states: Dict[str, UnitState],
    unit_map: Dict[str, WorkUnit],
    context_base: Optional[ProcessContext],
    originals: Dict[str, str],
) -> List[Tuple[str, ProcessResult]]:
    """
    自包含的 batch 处理，返回结果列表。

    和普通 unit 一样：
    - 有自己的 ID（基于内部 unit IDs）
    - 有自己的状态文件
    - Resume 通过 ID 匹配自然实现
    """
    mega_id = self._get_mega_unit_id(unit_ids)
    state_file = self._batch_states_dir / f"{mega_id}.json"

    results: List[Tuple[str, ProcessResult]] = []
    batch_requests = []
    units_to_process = {}

    # 1. 检查 Resume
    job_name = None
    if state_file.exists():
        state = self._load_batch_state(state_file)
        if state.job_state == "SUCCEEDED":
            # Job 已完成，获取结果
            return self._get_cached_batch_results(state.job_name, unit_ids, ...)
        elif state.job_state in ("PENDING", "RUNNING"):
            # Job 还在跑，继续等待
            job_name = state.job_name

    # 2. Pre-process 过滤 + 构建 requests（不是 resume 时）
    if job_name is None:
        for uid in unit_ids:
            unit = unit_map[uid]
            context = self._build_context(unit, context_base, ...)

            pre_result = self._hooks.pre_process(uid, unit.content, context)
            if not pre_result.should_process:
                # 直接返回 skip 结果
                results.append((uid, ProcessResult(
                    success=True,
                    skipped=True,
                    content=pre_result.fallback_result,
                    skip_reason=pre_result.skip_reason,
                )))
                continue

            prompt = self._processor.build_prompt(unit.content, context)
            contents = self._convert_prompt_to_batch_contents(prompt)
            batch_requests.append(BatchRequest(key=uid, contents=contents))
            units_to_process[uid] = (unit, context)

        # 3. 提交 Job
        if batch_requests:
            job_name = self._batch_client.submit(batch_requests)
            self._save_batch_state(state_file, job_name, "RUNNING")

    # 4. 等待完成
    if job_name:
        while True:
            job_info = self._batch_client.get_status(job_name)

            if job_info.state == BatchJobState.SUCCEEDED:
                break
            elif job_info.state in (BatchJobState.FAILED, BatchJobState.CANCELLED, BatchJobState.EXPIRED):
                # Job 级别失败：所有 unit 标记失败
                error = Exception(f"Batch job {job_info.state.name}: {job_info.error}")
                for uid in units_to_process:
                    results.append((uid, ProcessResult(success=False, error=error)))
                state_file.unlink(missing_ok=True)
                return results

            time.sleep(self._batch_poll_interval)

        # 5. 获取结果
        batch_responses = self._batch_client.get_results(job_name)
        response_map = {r.key: r for r in batch_responses}

        for uid, (unit, context) in units_to_process.items():
            resp = response_map.get(uid)
            if resp is None or resp.error:
                error_msg = resp.error if resp else "No response"
                results.append((uid, ProcessResult(success=False, error=Exception(str(error_msg)))))
            else:
                # 清理、验证（和 online 一样）
                cleaned = self._processor.clean_response(resp.text)
                original = originals.get(uid, unit.content)
                chapter_type = unit.chapter_type or ''
                transformed, hook_result = self._hooks.post_process(uid, original, cleaned, chapter_type, context)

                if hook_result.accepted:
                    final = self._processor.post_process(transformed, context)
                    results.append((uid, ProcessResult(
                        success=True,
                        content=final,
                        context_ready=hook_result.context_ready,
                        output_tokens=_count_tokens(final),
                    )))
                else:
                    results.append((uid, ProcessResult(
                        success=False,
                        error=Exception(hook_result.rejection_reason or "Validation failed"),
                    )))

        # 6. 清除状态文件
        state_file.unlink(missing_ok=True)

    return results
```

## Signal Handler

```python
def _cancel_all_batch_jobs(self):
    """Cancel 所有活跃的 batch job（用于 not resume 场景）"""
    if not self._batch_states_dir or not self._batch_states_dir.exists():
        return

    for state_file in self._batch_states_dir.glob("batch_*.json"):
        try:
            state = self._load_batch_state(state_file)
            if state and state.job_name:
                self._batch_client.cancel(state.job_name)
                logger.info(f"Cancelled batch job: {state.job_name}")
        except Exception as e:
            logger.warning(f"Failed to cancel batch job: {e}")
        finally:
            state_file.unlink(missing_ok=True)

def _handle_interrupt(self, signum, frame):
    """处理 SIGINT/SIGTERM"""
    logger.warning("Interrupt received, cancelling all batch jobs...")
    self._cancel_all_batch_jobs()

    # 恢复原始 handler 并重新抛出
    signal.signal(signal.SIGINT, self._original_sigint)
    signal.signal(signal.SIGTERM, self._original_sigterm)
    raise KeyboardInterrupt("Batch jobs cancelled by user")
```

## 代码变更总结

### 删除的代码

| 位置 | 代码 | 行数 |
|------|------|------|
| `execute()` | batch/online 初始分离逻辑 | ~50 行 |
| `execute()` | batch fallback 胶水代码（threshold 判断、retry、online fallback） | ~130 行 |
| `_process_batch()` | 独立的结果处理 for loop + 验证逻辑 | ~300 行 |
| `batch_state.py` | `processing_keys`, `batch_metadata`, `safety_blocked_keys` 等复杂字段 | ~50 行 |

**总计删除：~530 行**

### 新增的代码

| 位置 | 代码 | 行数 |
|------|------|------|
| `executor.py` | `_get_mega_unit_id()` | ~5 行 |
| `executor.py` | `_process_batch_as_unit()` | ~100 行 |
| `executor.py` | `_cancel_all_batch_jobs()` | ~15 行 |
| `executor.py` | 主循环 batch_futures 处理 | ~30 行 |
| `executor.py` | `_load_batch_state()`, `_save_batch_state()` | ~20 行 |

**总计新增：~170 行**

**净减少：~360 行**

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `executor.py` | 1. 重写主循环，统一 batch/online 处理<br>2. 新增 `_process_batch_as_unit()`<br>3. 新增 `_get_mega_unit_id()`<br>4. 简化 signal handler（遍历清理）<br>5. 删除 `_process_batch()` 的大部分代码 |
| `batch_state.py` | 大幅简化：只存储 `job_name` + `job_state` |
| `state.py` | 无需修改 |
| `pipeline_v2.py` | 添加 `batch_states_dir` 参数传递 |
| `factory_v2.py` | 创建 `batch_states/` 目录 |

## 验证步骤

### 1. Batch 验证失败 Requeue 测试
```bash
# 使用会产生验证失败的内容
uv run pdf2epub polish
# 验证：batch 验证失败后会 requeue，下一轮根据 chain[0] 决定继续 batch 或转 online
```

### 2. Resume 测试
```bash
# 启动处理
uv run pdf2epub polish
# Ctrl+C 中断

# Resume
uv run pdf2epub polish --resume
# 验证：
# - 相同 pending units → 相同 mega_id
# - 复用已有 batch job，不重新提交
# - 继续等待结果
```

### 3. Safety Error 混合测试
```bash
# 验证：
# - safety error 的 units：provider 被移除，chain[0] 变成 online:anthropic
# - truncation error 的 units：chain[0] 仍是 batch:gemini
# - 下一轮两种模式同时存在
```

### 4. 不够 Threshold 转 Online 测试
```bash
# 验证：
# - 第一轮 batch 后，只剩 3 个失败的 units（< 5）
# - 自动移除 batch entries，转 online 处理
```

### 5. 代码量对比
```bash
wc -l pdf2epub/core/executor/executor.py
# 重构前 ~1600 行，重构后 ~1250 行
```

## 不变式

1. **统一的 Requeue**：所有失败（batch/online）都调用 `_handle_result()`，失败自动 requeue
2. **Chain 驱动**：`state.chain[0].mode` 决定执行模式
3. **Quota 控制**：`can_retry()` 决定是否 requeue
4. **Mega Unit 局部性**：每个 mega unit 有独立的状态文件，不存在全局耦合
5. **ID 决定 Resume**：相同 pending units = 相同 mega_id = 自然 Resume
6. **收集阶段分流**：batch/online 的决定在收集阶段，threshold 不够自动转 online

## 设计原则

1. **Batch 是 Unit 的一种**：不是特殊的全局状态机，只是一个"大一点的 unit"
2. **状态局部化**：每个 mega unit 有自己的状态文件，和普通 unit 有自己的输出文件一样
3. **自然 Resume**：ID 由内容唯一确定，不需要特殊的匹配逻辑
4. **统一处理**：所有结果通过 `_handle_result()`，失败自然 requeue
5. **无胶水代码**：删除了所有 batch fallback/retry 特殊逻辑
6. **正交性**：batch vs online 只是 worker 任务的一种，不是特殊路径

## 与旧设计的对比

| 方面 | 旧设计 | 新设计 |
|------|--------|--------|
| Batch 概念 | 全局状态机 | Mega Unit（只是大一点的 unit） |
| 状态文件 | 单一 `batch_state.json` | 每个 mega unit 独立 `batch_{id}.json` |
| Resume 逻辑 | 复杂的 ID 匹配 + 特殊处理 | 自然的（相同 pending = 相同 ID） |
| 验证失败处理 | 直接标记 `validation_failed` | 通过 `_handle_result()` requeue |
| Fallback 逻辑 | ~130 行胶水代码 | 收集阶段自然判断 |
| 代码量 | ~1600 行 | ~1250 行 |
| 复杂度 | Batch/Online 二元对立 | 统一的 job queue |
