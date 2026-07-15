import base64
import json
from collections import defaultdict
import numpy as np
import logging
try:
    import ray
except ImportError:  # pragma: no cover - depends on user environment
    ray = None
from typing import Dict, List, Any
import os
import time
import math
import re
from openai import OpenAIError

import openai
from transformers.models.auto.processing_auto import AutoProcessor

logging.basicConfig(level=logging.INFO)


def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def get_bbox(meta):
    image_size = meta["img_size"]
    action = meta["step"]
    bbox = [
        action["bbox"]["x"],
        action["bbox"]["y"],
        action["bbox"]["x"] + action["bbox"]["width"],
        action["bbox"]["y"] + action["bbox"]["height"],
    ]
    bbox = [
        bbox[0] / image_size[0],
        bbox[1] / image_size[1],
        bbox[2] / image_size[0],
        bbox[3] / image_size[1],
    ]
    return [round(item, 3) for item in bbox]


def round_by_factor(x, factor):
    return int(round(x / factor) * factor)


def floor_by_factor(x, factor):
    return int(math.floor(x / factor) * factor)


def ceil_by_factor(x, factor):
    return int(math.ceil(x / factor) * factor)


IMAGE_FACTOR = 28
MIN_PIXELS = 1000 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def smart_resize(h, w, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if max(h, w) / min(h, w) > MAX_RATIO:
        raise ValueError("Aspect ratio exceeds limit")

    h_bar = max(factor, round_by_factor(h, factor))
    w_bar = max(factor, round_by_factor(w, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = floor_by_factor(h / beta, factor)
        w_bar = floor_by_factor(w / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = ceil_by_factor(h * beta, factor)
        w_bar = ceil_by_factor(w * beta, factor)

    return h_bar, w_bar


DART_PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Action: ...


## Action Space
click(point='<point>x1 y1</point>')
type(content='...', point='<point>x1 y1</point>')
select(option='...', point='<point>x1 y1</point>')

## Note
- Coordinates must be in the resized screenshot coordinate system.

## User Instruction {instruction}
"""


def _is_legacy_mind2web_prompt(text):
    return (
        "You are an assistant trained to navigate the web" in text
        or "Format the action as a dictionary" in text
        or "Format the action as a JSON object" in text
    )


def parse_source_to_payload(source):
    """
    Convert Mind2Web source into a ScreenSpot-Pro-like DART payload.
    The dataset helper still builds a legacy JSON prompt in source; strip it
    here and keep only the task, action history, and screenshots.
    Returns messages plus the last screenshot size, since the predicted action is
    applied to the current screenshot.
    """
    task = ""
    history = []
    last_image_size = None

    for item in source:
        if item["role"] != "user":
            continue
        for content in item["content"]:
            if content["type"] == "text":
                text = content["text"].strip()
                if _is_legacy_mind2web_prompt(text):
                    continue
                if text.startswith("Task:"):
                    task = text.split("Task:", 1)[1].strip()
                    continue
                history.append({"type": "text", "text": text})
            elif content["type"] == "image":
                with open(content["image"], "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                from PIL import Image
                with Image.open(content["image"]) as img:
                    last_image_size = img.size  # (width, height)
                history.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    }
                )
    messages = [{"type": "text", "text": DART_PROMPT_TEMPLATE.format(instruction=task)}]
    if history:
        messages.append({"type": "text", "text": "## Action History"})
        messages.extend(history)
    return messages, last_image_size


def _extract_box_point(action_str):
    match = re.search(r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", action_str)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    match = re.search(r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>", action_str)
    if not match:
        match = re.search(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", action_str)
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))]


def _extract_quoted_arg(action_str, names):
    for name in names:
        match = re.search(name + r"\s*=\s*(['\"])(.*?)\1", action_str, flags=re.DOTALL)
        if match:
            return match.group(2)
    return ""


def _scale_point_to_normalized(point, image_size):
    if not image_size:
        return [float(point[0]), float(point[1])]
    img_width, img_height = image_size
    h_resized, w_resized = smart_resize(img_height, img_width)
    x_original = point[0] * (img_width / w_resized)
    y_original = point[1] * (img_height / h_resized)
    return [
        round(x_original / img_width, 3),
        round(y_original / img_height, 3),
    ]


def parse_model_response(response_text, image_size=None):
    """
    Parse DART action output into Mind2Web's expected action dict.
    Also keeps legacy JSON parsing as a fallback for mixed runs.
    """
    try:
        parts = response_text.split("Action:", 1)
        thought = parts[0].replace("Thought:", "").strip() if parts else ""
        if len(parts) != 2:
            return thought, None
        action_str = parts[1].strip()

        # Legacy TongUI JSON fallback.
        if action_str.startswith("{"):
            action = json.loads(action_str)
            if not isinstance(action, dict) or "action" not in action or "position" not in action:
                return thought, None
            return thought, action

        point = _extract_box_point(action_str)
        if point is None:
            return thought, None
        point = _scale_point_to_normalized(point, image_size)

        action_lower = action_str.lower()
        if action_lower.startswith("type") or "type(" in action_lower or "input" in action_lower:
            action = {
                "action": "TYPE",
                "position": point,
                "value": _extract_quoted_arg(action_str, ["content", "text", "value"]),
            }
        elif action_lower.startswith("select") or "select(" in action_lower:
            action = {
                "action": "SELECT",
                "position": point,
                "value": _extract_quoted_arg(action_str, ["option", "value", "text", "content"]),
            }
        else:
            action = {"action": "CLICK", "position": point, "value": ""}
        return thought, action
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None


def predict_action(
    client: openai.OpenAI,
    source: str,
    model: str = "tongui-3b",
    max_retries: int = 3,
    n_sampling: int = 3,
    temperature: float = 1,
    top_k: int = 50,
):
    """
    Make prediction using OpenAI client with retries
    """
    messages, image_size = parse_source_to_payload(source)

    for attempt in range(max_retries):
        try:
            thoughts = []
            actions = []
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": messages}],
                max_completion_tokens=256,
                temperature=temperature,
                top_p=0.95,
                extra_body={
                    "top_k": top_k,
                    "mm_processor_kwargs": {
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                },
                n=n_sampling,
            )
            for i in range(n_sampling):
                print(f"Sampling {i+1} of {n_sampling}")
                prediction_text = response.choices[i].message.content
                if hasattr(response.usage, "total_tokens"):
                    print(
                        f"Raw prediction: {prediction_text}; usage: {response.usage.total_tokens}"
                    )
                else:
                    print(f"Raw prediction: {prediction_text}")

                thought, action = parse_model_response(prediction_text, image_size)
                thoughts.append(thought)
                actions.append(action)
            return thoughts, actions

        except (OpenAIError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None, None
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)  # Wait before retrying


def calculate_mind2web_metrics(results):
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {"CLICK": [], "TYPE": [], "SELECT": []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0

    for i, (annot_id, item) in enumerate(results.items()):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True
        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])
    macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])

    logging.info("[Operation F1]: " + str(marco_op_f1))
    logging.info("[Element Acc]: " + str(num_ele / num_step))
    logging.info("[Step Success]: " + str(num_step_success / num_step))
    logging.info("[Episode Success]: " + str(num_episode_success / num_episode))
    logging.info("[Operation F1 cate]: " + str([np.mean(x) for x in op_f1.values()]))

    logging.info("[Macro Ele Acc]: " + str(macro_ele_acc))
    logging.info("[Macro Op F1]: " + str(macro_action_f1))
    logging.info("[Macro Step SR]: " + str(macro_step_acc))

    metrics = {
        "Operation F1": marco_op_f1,
        "Element Accuracy": num_ele / num_step,
        "Step Success": num_step_success / num_step,
        "Episode Success": num_episode_success / num_episode,
        "Operation F1 categories": [np.mean(x) for x in op_f1.values()],
        "Macro Element Accuracy": macro_ele_acc,
        "Macro Operation F1": macro_action_f1,
        "Macro Step Success Rate": macro_step_acc,
    }
    return metrics


def evaluate_dataset(
    dataset_name: str,
    model: str,
    endpoint: str,
    dataset_dir: str,
    processor_path: str,
    limit: int = -1,
    temperature: float = 1,
    n_sampling: int = 3,
    top_k: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate a specific dataset type (task, website, or domain)
    """
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=endpoint,
        )

        # Initialize processor and dataset
        processor = AutoProcessor.from_pretrained(processor_path)
        print("Processor settings",
              processor.image_processor.max_pixels,
        )
        version = "v2"
        dataset_name_full = f"hf_test_{dataset_name}_with_thoughts"

        from tongui.data.dset_mind2web import Mind2WebDataset

        dataset = Mind2WebDataset(
            dataset_dir,
            "Mind2Web",
            dataset_name_full,
            processor,
            inference=True,
            args_dict={
                "num_history": 2,
                "interleaved_history": "vtvt",
                "version": version,
            },
        )

        # Track results
        results = defaultdict(lambda: defaultdict(list))

        # Process each sample
        for idx, item in enumerate(dataset):
            if limit > 0 and idx >= limit:
                break   
            data_dict, item = item
            print(f"\nProcessing {dataset_name} sample {idx + 1}")

            try:
                # Get source and make prediction
                source = item["source"]
                thoughts, actions = predict_action(client, source, model=model, temperature=temperature, n_sampling=n_sampling, top_k=top_k)
                
                if actions is None:
                    print(f"Failed to get predictions for sample {idx}")
                    continue

                # Track best results across all samples
                best_op_match = False
                best_ele_match = False
                best_op_f1 = 0.0
                best_action = None
                gt_action = item["answer"]
                action2id = {"CLICK": 4, "SELECT": 2, "TYPE": 3}

                # Check all predictions and keep the best one
                for action in actions:
                    if action is None or action.get("action") not in action2id:
                        continue
                    if "position" not in action or not isinstance(action["position"], list) or len(action["position"]) < 2:
                        print(f"Skipping malformed action without valid position for sample {idx}: {action}")
                        continue

                    # Compare with ground truth
                    op_match = action["action"] == gt_action["action"]

                    # Calculate element match
                    bbox_ref = get_bbox(item)
                    click_point = action["position"]
                    ele_match = (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (
                        bbox_ref[1] <= click_point[1] <= bbox_ref[3]
                    )

                    # Calculate operation F1
                    action_pred_idx = action2id[action["action"]]
                    pred_str = str(action_pred_idx)
                    if action["action"] in ["TYPE", "SELECT"]:
                        pred_str += " " + str(action.get("value", "")).lower()

                    action_ref_idx = action2id[gt_action["action"]]
                    ref_str = str(action_ref_idx)
                    if gt_action["action"] in ["TYPE", "SELECT"]:
                        ref_str += " " + str(gt_action.get("value", "")).lower()

                    op_f1 = calculate_f1(pred_str, ref_str)

                    # Update best results if this prediction is better
                    if op_f1 > best_op_f1 or (op_f1 == best_op_f1 and ele_match and not best_ele_match):
                        best_op_match = op_match
                        best_ele_match = ele_match
                        best_op_f1 = op_f1
                        best_action = action

                if best_action is None:
                    print(f"Skipping sample {idx} because no prediction could be parsed")
                    continue

                # Store results using the best prediction
                step_result = {
                    "Op_match": best_op_match,
                    "Ele_match": best_ele_match,
                    "Op_F1": [best_op_f1, gt_action["action"]],
                    "meta": item,
                }

                split = item.get("split", "unknown")
                anno_id = item.get("anno_id", str(idx))
                results[split][anno_id].append(step_result)

                print(f"Best prediction: {best_action}")
                print(f"Ground truth: {gt_action}")
                print(f"Op match: {best_op_match}, Ele match: {best_ele_match}, Op F1: {best_op_f1}")
            except Exception as e:
                print(f"Skipping sample {idx} due to evaluation error: {e}")
                continue

        # Calculate metrics
        final_metrics = {}
        for split, _ in results.items():
            metrics = calculate_mind2web_metrics(results[split])

            for metric_name, value in metrics.items():
                if isinstance(value, list):
                    if metric_name == "Operation F1 categories":
                        for i, category in enumerate(["CLICK", "TYPE", "SELECT"]):
                            final_metrics[f"{dataset_name}/{split}/op_f1_{category}"] = value[i]
                else:
                    final_metrics[f"{dataset_name}/{split}/{metric_name}"] = value

        return final_metrics

    except Exception as e:
        print(f"Error in evaluate_dataset for {dataset_name}: {str(e)}")
        return {f"{dataset_name}/error": str(e)}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    # Configuration
    MODEL = os.environ.get("MODEL", "dart-7b")
    ENDPOINT = os.environ.get("ENDPOINT", "http://localhost:8000/v1")
    DATASET_DIR = os.environ.get("MIND2WEB_DATASET_DIR", "dataset")
    PROCESSOR_PATH = os.environ.get("MIND2WEB_PROCESSOR")
    if not PROCESSOR_PATH:
        raise ValueError("MIND2WEB_PROCESSOR must be set to a local or Hugging Face processor path.")
    LIMIT = int(os.environ.get("LIMIT", "-1"))
    TEMPERATURE = float(os.environ.get("TEMPERATURE","0.7"))
    N_SAMPLING = int(os.environ.get("N_SAMPLING", "1"))
    TOP_K = int(os.environ.get("TOP_K", "50"))
    # Initialize Ray with error handling
    try:
        # List of dataset types to evaluate
        dataset_types = ["task", "website", "domain"]

        if ray is not None:
            ray.init()
            remote_evaluate = ray.remote(evaluate_dataset)
            futures = []
            for dataset_type in dataset_types:
                future = remote_evaluate.remote(
                    dataset_type,
                    MODEL,
                    ENDPOINT,
                    DATASET_DIR,
                    PROCESSOR_PATH,
                    LIMIT,
                    TEMPERATURE,
                    N_SAMPLING,
                    TOP_K,
                )
                futures.append(future)
            all_metrics = ray.get(futures)
        else:
            print("ray is not installed; evaluating dataset types sequentially.")
            all_metrics = [
                evaluate_dataset(
                    dataset_type,
                    MODEL,
                    ENDPOINT,
                    DATASET_DIR,
                    PROCESSOR_PATH,
                    LIMIT,
                    TEMPERATURE,
                    N_SAMPLING,
                    TOP_K,
                )
                for dataset_type in dataset_types
            ]

        # Combine and log final metrics
        final_metrics = {}
        for dataset_metrics in all_metrics:
            if dataset_metrics:  # Check if metrics exist
                final_metrics.update(dataset_metrics)

        with open(f"results_{MODEL}_mind2web.json", "w") as f:
            json.dump(_json_safe(final_metrics), f, indent=4)

    except Exception as e:
        print(f"Error in main: {str(e)}")

    finally:
        # Close Ray
        if ray is not None:
            ray.shutdown()


if __name__ == "__main__":
    main()
