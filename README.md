# DART ScreenSpot 评测说明

这个目录包含基于 vLLM / OpenAI API 格式的 DART GUI grounding 评测脚本，目前主要用于：

- ScreenSpot-v2
- ScreenSpot-Pro

## 1. 安装依赖

先准备 Python 环境，然后安装基础依赖：

```bash
pip install openai pillow tqdm
```

如果你要打开可视化，也就是把脚本里的 `DISPLAY_IMAGES = True`，还需要额外安装：

```bash
pip install matplotlib numpy
```

默认 `DISPLAY_IMAGES = False`，所以正常评测不需要 `matplotlib` 和 `numpy`。

## 2. 放置数据

建议从 `dart_eval` 目录下运行脚本：

```bash
cd /path/to/dart_eval
```

当前脚本默认读取相对路径 `dataset`，因此数据需要放成下面的结构：

```text
dart_eval/
  eval/
    run_screenspot_2_vllm_dart.py
    run_screenspot_pro_vllm_dart.py
  dataset/
    ScreenSpot-v2/
      *.json
      screenspotv2_image/
        ...
    ScreenSpot-Pro/
      annotations/
        *.json
      images/
        ...
```

标注字段要求：

- ScreenSpot-v2：每条样本需要包含 `img_filename`、`instruction`、`bbox`，可选 `data_type`。
- ScreenSpot-Pro：每条样本需要包含 `img_filename`、`instruction`、`bbox`，可选 `group` / `ui_type`。

脚本会把模型预测点映射回原图坐标，然后和 `bbox` 做命中判断。

模型输出需要包含下面这种 DART action 格式：

```text
Action: click(point='<point>x y</point>')
```

其中 `x y` 是模型处理后的 resized image 坐标。脚本会根据图片原始尺寸和 `smart_resize()` 规则，把它还原到原图坐标后再计算准确率。

## 3. 配置模型服务

需要先启动一个兼容 OpenAI API 的 vLLM 服务。例如：

```bash
vllm serve /path/to/your/model \
  --served-model-name dart-7b \
  --host 0.0.0.0 \
  --port 8000
```

然后在要运行的脚本里修改：

```python
MODEL = "dart-7b"
ENDPOINT = "http://your-host:8000/v1"
```

注意：`MODEL` 必须和 vLLM 暴露出来的模型名一致，通常就是 `--served-model-name` 的值。

## 4. 运行评测

ScreenSpot-v2：

```bash
cd /path/to/dart_eval
python eval/run_screenspot_2_vllm_dart.py
```

ScreenSpot-Pro：

```bash
cd /path/to/dart_eval
python eval/run_screenspot_pro_vllm_dart.py
```

运行结束后，结果 JSON 会保存在当前工作目录：

```text
results_dart-7b_screenspot_2.json
results_dart-7b_screenspot_pro.json
```

同时脚本也会在终端打印按 split / data type 分组的准确率。

## 5. 并发评测

脚本默认会并发请求 vLLM，默认并发数是 `8`。

可以通过环境变量调整并发数，不需要改代码：

```bash
SCREENSPOT_MAX_WORKERS=4 python eval/run_screenspot_2_vllm_dart.py
SCREENSPOT_MAX_WORKERS=8 python eval/run_screenspot_2_vllm_dart.py
SCREENSPOT_MAX_WORKERS=16 python eval/run_screenspot_2_vllm_dart.py
```

并发数不是越大越快。如果 vLLM 已经打满、显存压力变大或者请求开始超时，就把 `SCREENSPOT_MAX_WORKERS` 调小。

## 6. 常用配置项

每个脚本的 `main()` 里都有这些配置：

```python
DISPLAY_IMAGES = False
LIMIT = -1
MODEL = "dart-7b"
ENDPOINT = "http://..."
FILTER_BY_SPLIT = None
```

含义：

- `DISPLAY_IMAGES`：是否保存预测点可视化图片。默认关闭。
- `LIMIT`：限制评测样本数。`-1` 表示全量评测；调试时可以设成 `5` 或 `10`。
- `MODEL`：请求 vLLM 时使用的模型名。
- `ENDPOINT`：OpenAI-compatible vLLM 服务地址。
- `FILTER_BY_SPLIT`：当前脚本里基本未启用，可以先保持 `None`。

## 7. 快速测试

第一次跑建议先把脚本里的：

```python
LIMIT = -1
```

改成：

```python
LIMIT = 5
```

然后运行一个脚本，确认：

- 能正常读取标注和图片。
- vLLM endpoint 能返回结果。
- 模型输出里有 `Action: click(point='<point>x y</point>')`。
- 终端能打印 `Predicted point` 和准确率统计。

确认没问题后，再把 `LIMIT` 改回 `-1` 跑完整评测。
