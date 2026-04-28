# MATLAB Job 提交流程（每次 2 个 job）

## 文件

- `tmp/matlab_job_test_sge.sh`：SGE job 脚本，采用 `-t 1-2` 和 `-tc 2`，每次提交运行 2 个 task（2 个 job）。
  - 若 `matlab` 不在 PATH，会先尝试 `module load matlab/R2021a`。
- `tmp/submit_matlab_job_test.sh`：提交脚本，调用 `qsub` 提交上述 job 脚本。

## 使用

```bash
bash /old_Users/yojiao/hipp/tmp/submit_matlab_job_test.sh
```

## 结果判定

- 输出日志：`tmp/matlab_job_test/results/job_<JOB_ID>_task_<TASK_ID>.log`
- 成功标记：`tmp/matlab_job_test/results/job_<JOB_ID>_task_<TASK_ID>.ok`
- 两个 task 都生成 `.ok` 文件，说明本轮 MATLAB 提交测试跑通。
