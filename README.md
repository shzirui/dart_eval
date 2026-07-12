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

## 8. 合并 safetensors 模型

仓库里还提供了一个 Hugging Face safetensors 权重线性合并工具：

```text
tools/merge_hf_safetensors.py
```

额外依赖：

```bash
pip install torch safetensors
```

示例：

```bash
python tools/merge_hf_safetensors.py \
  --model-a models/dart-gui-7b \
  --model-b models/UI-TARS-1.5-7B \
  --output models/dart-gui-uitars-0.5-0.5 \
  --weight-a 0.5 \
  --weight-b 0.5
```

这个脚本会逐 shard 合并 `model.safetensors.index.json` 对应的权重，不会一次性把完整模型加载到内存。非权重文件，例如 tokenizer、config 等，会从 `--copy-from` 指定的模型目录复制，默认从 `model-a` 复制。

常用参数：

- `--weight-a` / `--weight-b`：两个模型的线性合并权重。
- `--copy-from a|b`：非权重文件从哪个模型目录复制。
- `--compute-dtype`：合并时临时计算 dtype，可选 `float32`、`bfloat16`、`float16`。
- `--output-dtype`：输出权重 dtype，默认 `source`，即保持 `model-a` 的原始 dtype。
- `--dry-run`：只检查模型结构和 shard 是否兼容，不实际写输出。
- `--force`：输出目录已存在时强制覆盖。

## 9. Mind2Web 评测

仓库里也包含 DART 版本的 Mind2Web vLLM 评测脚本：

```bash
python eval/run_mind2web_vllm_dart.py
```

Mind2Web 需要额外依赖：

```bash
pip install numpy openai pillow torch transformers qwen-vl-utils
```

可选依赖：

```bash
pip install ray
```

- 没有 `ray` 时，脚本会顺序评测 `task`、`website`、`domain` 三个 split。

数据默认也从相对路径 `dataset` 读取，目录结构如下：

```text
dart_eval/
  dataset/
    Mind2Web/
      metadata/
        hf_test_task_with_thoughts.json
        hf_test_website_with_thoughts.json
        hf_test_domain_with_thoughts.json
      ming2web_images/
        ...
```

推荐通过环境变量配置模型、服务地址、数据路径和 processor：

```bash
MODEL=dart-7b \
ENDPOINT=http://localhost:8000/v1 \
MIND2WEB_DATASET_DIR=dataset \
MIND2WEB_PROCESSOR=/path/to/processor_or_model \
LIMIT=100 \
python eval/run_mind2web_vllm_dart.py
```

配置项说明：

- `MODEL`：vLLM 暴露出来的模型名。
- `ENDPOINT`：OpenAI-compatible vLLM 服务地址。
- `MIND2WEB_DATASET_DIR`：数据根目录，默认 `dataset`。
- `MIND2WEB_PROCESSOR`：必填，用于构造 Mind2Web prompt 和图片输入的 Hugging Face processor。可以填本地模型目录，也可以填 Hugging Face repo id。
- `LIMIT`：每个 split 评测多少条，默认 `100`；设为 `-1` 表示全量。
- `N_SAMPLING`：每条样本采样次数，默认 `1`。
- `TOP_K`：采样参数，默认 `50`。

输出结果会保存为：

```text
results_dart-7b_mind2web.json
```
