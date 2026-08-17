# Qwen3-8B 本地输出实现计划

> **供执行代理使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行，并用复选框跟踪进度。

**目标：** 让 Qwen3-8B GPT4All 启动器默认把输出、日志和 checkpoint 写入本地磁盘，并拒绝 NFS 输出路径。

**架构：** 代码、模型和数据路径仍位于仓库目录。增加可覆盖的绝对路径 `OUTPUT_ROOT`，实验输出和日志目录均从它派生，并在启动 Python 前检查文件系统。

**技术栈：** Bash、`findmnt`、`df`、Python/pytest 静态启动器测试

---

### 任务 1：用失败测试约束本地输出行为

**文件：**
- 修改：`tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **步骤 1：增加默认路径和 NFS 防护测试**

```python
def test_launcher_defaults_outputs_and_logs_to_local_storage():
    text = launcher_text()
    assert (
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/'
        'JudgeStealer_outputs"' in text
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text
    assert 'OUT="$OUTPUT_ROOT/$NAME"' in text
    assert 'LOG_ROOT="$OUTPUT_ROOT/qwen3_8b_gpt4all_gpt5_four_stage_logs"' in text


def test_launcher_reports_storage_and_rejects_nfs_outputs():
    text = launcher_text()
    assert 'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"' in text
    assert 'df -hP "$OUTPUT_ROOT"' in text
    assert 'nfs|nfs4)' in text
    assert 'ERROR network filesystem output is not allowed' in text
```

- [ ] **步骤 2：运行新测试，确认处于 RED 状态**

运行：

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：两个新测试失败，因为启动器仍从 `$ROOT/outputs` 派生 `OUT` 和 `LOG_ROOT`，且尚未实现文件系统预检查。

- [ ] **步骤 3：提交失败测试**

```bash
git add tests/test_qwen3_8b_gpt4all_launcher.py
git commit -m "Test local output guard for Qwen3-8B launcher"
```

### 任务 2：将写入路径切换到本地磁盘

**文件：**
- 修改：`launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- 测试：`tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **步骤 1：从 `OUTPUT_ROOT` 派生所有可写路径**

将现有的 `OUT` 和 `LOG_ROOT` 定义替换为：

```bash
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
OUT="$OUTPUT_ROOT/$NAME"
LOG_ROOT="$OUTPUT_ROOT/qwen3_8b_gpt4all_gpt5_four_stage_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
LOG="$LOG_ROOT/$NAME.log"
```

- [ ] **步骤 2：在 `log_status` 后增加文件系统预检查**

```bash
check_output_storage() {
  local fs_type=""
  local available="unknown"

  if command -v findmnt >/dev/null 2>&1; then
    fs_type="$(findmnt -n -o FSTYPE -T "$OUTPUT_ROOT" 2>/dev/null || true)"
  else
    log_status "WARNING findmnt unavailable; output filesystem type not checked"
  fi
  available="$(df -hP "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -n "$available" ]] || available="unknown"
  [[ -n "$fs_type" ]] || fs_type="unknown"

  log_status "STORAGE output_root=$OUTPUT_ROOT fstype=$fs_type available=$available"
  case "$fs_type" in
    nfs|nfs4)
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"
      exit 1
      ;;
  esac
}

check_output_storage
```

保留位于此函数调用之前的 `mkdir -p "$LOG_ROOT"`，使默认目录在 `findmnt` 和 `df` 检查前已经存在。

- [ ] **步骤 3：运行启动器测试，确认处于 GREEN 状态**

运行：

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：`6 passed`。

- [ ] **步骤 4：验证 Bash 语法**

运行：

```bash
bash -n launch_qwen3_8b_gpt4all_gpt5_four_stage.sh
```

预期：退出状态为 0，且没有输出。

- [ ] **步骤 5：检查聚焦后的差异**

运行：

```bash
git diff --check
git diff -- launch_qwen3_8b_gpt4all_gpt5_four_stage.sh tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：只有本地输出路由、存储预检查及其测试发生变化。

- [ ] **步骤 6：提交实现**

```bash
git add launch_qwen3_8b_gpt4all_gpt5_four_stage.sh tests/test_qwen3_8b_gpt4all_launcher.py
git commit -m "Write Qwen3-8B experiment outputs to local storage"
```

### 任务 3：最终验证与推送

**文件：**
- 验证：`launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- 验证：`tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **步骤 1：运行聚焦回归测试**

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py tests/test_skywork_dataset.py
```

预期：所有聚焦测试通过。

- [ ] **步骤 2：确认用户无关文件未被改动**

```bash
git status --short
```

预期：只保留原先未跟踪的 `PROJECT_MEMORY.md`、`WORK_LOG.md` 和 `launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh`。

- [ ] **步骤 3：推送提交**

```bash
git push origin main
```

预期：GitHub 上的 `main` 已更新，且三个无关的未跟踪文件没有被改动。
