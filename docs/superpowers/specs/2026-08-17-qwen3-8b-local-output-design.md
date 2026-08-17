# Qwen3-8B GPT4All 启动器本地输出设计

## 目标

修改 `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`，避免训练结果、日志和
checkpoint 继续写入 `/data` 的 NFSv3 文件系统。模型、代码和数据仍从仓库目录
读取，训练写入改到服务器本地磁盘。

## 输出路径

- 默认输出根目录：
  `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs`
- 允许通过 `OUTPUT_ROOT` 环境变量覆盖默认值。
- 实验输出目录和日志目录都必须位于 `OUTPUT_ROOT` 下。
- 不改变实验名称、模型路径、数据路径或训练参数。

示例：

```bash
./launch_qwen3_8b_gpt4all_gpt5_four_stage.sh 3

OUTPUT_ROOT=/root/JudgeStealer_outputs \
  ./launch_qwen3_8b_gpt4all_gpt5_four_stage.sh 3
```

## 启动前检查

启动器先创建 `OUTPUT_ROOT` 和日志目录，然后检查输出路径所在的文件系统：

- 使用 `findmnt` 获取文件系统类型，并把类型和可用空间写入状态日志。
- 若类型为 `nfs` 或 `nfs4`，立即退出并记录明确错误。
- 若 `findmnt` 不可用或无法识别类型，记录警告但不阻止启动。
- 保留现有的完成、重复进程和不完整输出保护。

该检查只约束输出路径；仓库、模型和数据可以继续位于 NFS，因为训练只读取它们。

## 数据保存边界

启动器不自动把结果同步回 `/data`，也不自动删除本地结果。训练成功后由用户执行
单独的 `rsync --checksum`，避免启动器在训练结束时引入长时间网络写入或误覆盖。

## 测试

扩展现有静态启动器测试，验证：

- 默认 `OUTPUT_ROOT` 指向约定的本地目录。
- `OUT` 和 `LOG_ROOT` 均从 `OUTPUT_ROOT` 派生。
- 启动器检测并拒绝 `nfs` 与 `nfs4`。
- 原有模型、数据、四阶段训练和 selector 配置保持不变。

不运行完整训练测试。
