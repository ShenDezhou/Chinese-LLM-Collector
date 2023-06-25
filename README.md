# Collector:LLM驱动的任务执行者

本文提出一种基于LLM的任务生成方法，可以完成开放世界探索任务。以收集LLM中文数据集为目标，评估LLM对于环境认知条件下的任务提出能力。

## LLM的系统设定

我们为`执行者`设置的角色设定为`助手`，并为其设计终极目标`发现尽可能多的多样数据集`。
在系统设计中给出了下面四个组件：角色定义、信息模版、生成准则、生成格式示例。
```text
You are a helpful assistant that tells me the next immediate task to do in Chinese-LLM-Collector. My ultimate goal is to discover as many diverse datasets as possible, acquire as many diverse datasets as possible and become the best Chinese LLM Datasets Collector in the world.

I will give you the following information:
Downloaded datasets so far: ...
Downloaded subsets so far: ...
Downloaded dataset parquets so far: ...
Completed tasks so far: ...
Failed tasks that are too hard: ...
Task: ...
Dataset: ...
Subsets: ...
Working subset: ...
Splits: ...
Split: ...

You must follow the following criteria:
1) You should act as a mentor and guide me to the next dataset based on my current learning progress.
2) The next tasks should follow a concise format, such as "Collect [dataset] [subset] [split]".
3) Enumerate all the combination as tasks. [dataset] can select from Dataset information. [subset] can be selected from Subsets information. [split] can be selected from Splits information.
4) Do remove collected subset tasks. If Downloaded subsets contained the subset, do not include it.

You should only respond in the format as described below:
RESPONSE FORMAT:
Reasoning: Based on the information provided, you have already collected the clue afqmc dataset.
Task: Collect clue tnews train.
Task: Collect clue tnews validation.
Task: Collect clue tnews test.
Task: Collect clue iflytek train.
Task: Collect clue iflytek validation.
Task: Collect clue iflytek test.
...
```

用户区数据为环境信息（nodejs服务器上已经下载的数据集及其附属信息）、已成功任务、已失败任务、当前目标数据集及其附属信息。
```text

Downloaded datasets so far: c3, clue, dicache, lccc

Downloaded subsets so far: c3/mixed, clue/tnews, clue/afqmc, dicache/the, dicache/the/dataset_infos.json, dicache/lccc, dicache/lccc/dataset_infos.json, dicache/clue, dicache/clue/dataset_infos.json, dicache/c3, dicache/c3/dataset_infos.json, lccc/base

Downloaded dataset parquets so far: c3/mixed/c3-test.parquet, clue/tnews/clue-validation.parquet, clue/tnews/clue-train.parquet, clue/tnews/clue-test.parquet, clue/afqmc/clue-validation.parquet, clue/afqmc/clue-train.parquet, clue/afqmc/clue-test.parquet, lccc/base/lccc-test.parquet

Completed tasks so far: Collect clue afqmc train, Collect clue afqmc validation, Collect clue afqmc test


Failed tasks that are too hard: None


Task: clue afqmc train
Dataset: clue
Subsets: afqmc, tnews, iflytek, cmnli, cluewsc2020, csl, cmrc2018, drcd, chid, c3, ocnli, diagnostics
Working subset: afqmc
Splits: Train, Validation, Test
Split: train


```

## 试验

在gpt-3.5-turbo-0613模型下，成功生成任务数据。
```text
Reasoning: Based on the information provided, you have already collected the clue afqmc dataset.

Task: Collect clue tnews train.
Task: Collect clue tnews validation.
Task: Collect clue tnews test.
Task: Collect clue iflytek train.
Task: Collect clue iflytek validation.
Task: Collect clue iflytek test.
Task: Collect clue cmnli train.
Task: Collect clue cmnli validation.
Task: Collect clue cmnli test.
Task: Collect clue cluewsc2020 train.
Task: Collect clue cluewsc2020 validation.
Task: Collect clue cluewsc2020 test.
Task: Collect clue csl train.
Task: Collect clue csl validation.
Task: Collect clue csl test.
Task: Collect clue cmrc2018 train.
Task: Collect clue cmrc2018 validation.
Task: Collect clue cmrc2018 test.
Task: Collect clue drcd train.
Task: Collect clue drcd validation.
Task: Collect clue drcd test.
Task: Collect clue chid train.
Task: Collect clue chid validation.
Task: Collect clue chid test.
Task: Collect clue c3 train.
Task: Collect clue c3 validation.
Task: Collect clue c3 test.
Task: Collect clue ocnli train.
Task: Collect clue ocnli validation.
Task: Collect clue ocnli test.
Task: Collect clue diagnostics train.
Task: Collect clue diagnostics validation.
Task: Collect clue diagnostics test.
```

由生成数据可知，ChatGPT可以较好理解已完成任务(Completed tasks)，并且遵照执行了生成约束中的2)、3)、4)条内容，并且生成任务全部可执行。

LLM交互完整文本日志可见[llm-curri_20230625_165812](event_log%2Fllm-curri_20230625_165812)、[llm-task_20230625_165942](event_log%2Fllm-task_20230625_165942)。
Python运行日志可见[main_20230625_1658.log](event_log%2Fmain_20230625_1658.log)

# 结论

本文设计了一个LLM驱动的数据收集者，基于[Chinese-DuckDB](https://github.com/ShenDezhou/Chinese-DuckDB)模拟环境，数据收集者收集到了clue数据集下全部子集数据，
完成了开放世界探索任务。本项目的意义在于，仅调用一次ChatGPT，即实现了多个子任务的生成，节省了API调用次数。
