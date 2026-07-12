import base64
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import openai
from openai import OpenAIError
from PIL import Image
from tqdm import tqdm


IMAGE_FACTOR = 28
MIN_PIXELS = 1000 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

TAP_DISTANCE_THRESHOLD = 0.14
SWIPE_DISTANCE_THRESHOLD = 0.04
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4


AITW_PROMPT = """You are a GUI agent. You are given a mobile task, action history, and screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
click(point='<point>x y</point>')
type(content='...')
scroll(direction='up|down|left|right')
press_back()
press_home()
press_enter()
status_task_complete()
status_task_impossible()

## Note
- Use English in `Thought` part.
- For click, coordinates must be in the resized current screenshot coordinate system.
- Use only one action.
"""


def round_by_factor(x, factor):
    return int(round(x / factor) * factor)


def floor_by_factor(x, factor):
    return int(math.floor(x / factor) * factor)


def ceil_by_factor(x, factor):
    return int(math.ceil(x / factor) * factor)


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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def image_content(image_path):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
    }


def get_answer(step):
    action_type_id = step["action_type_id"]
    action_type_text = step["action_type_text"]

    click_point = None
    type_text = None
    if action_type_id == 4 and action_type_text == "click":
        touch = step["touch"]
        lift = step["lift"]
        click_point = [(touch[0] + lift[0]) / 2, (touch[1] + lift[1]) / 2]
        click_point = [round(item, 2) for item in click_point]
    elif action_type_id == 3:
        type_text = step["type_text"]

    return {"action": action_type_text.upper(), "value": type_text, "position": click_point}


def load_aitw_data(dataset_dir, version="v2", limit=-1):
    base_dir = os.path.join(dataset_dir, "AITW")
    meta_dir = os.path.join(base_dir, "metadata")
    img_dir = os.path.join(base_dir, "aitw_images")
    dataset_name = "hf_test_with_thoughts" if version == "v2" else "hf_test"
    meta_path = os.path.join(meta_dir, f"{dataset_name}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    print(f"Dataset: AITW; Split: {dataset_name}; # samples: {len(data)}")
    return data, img_dir


def resolve_image_path(img_dir, img_url):
    if img_url.endswith(".png"):
        return os.path.join(img_dir, img_url)
    return os.path.join(img_dir, f"{img_url}.png")


def build_messages(item, img_dir, num_history):
    current_image_path = resolve_image_path(img_dir, item["img_url"])
    content = [{"type": "text", "text": AITW_PROMPT}]
    content.append({"type": "text", "text": f"Task: {item['task']}"})

    history = item.get("step_history", [])[-num_history:] if num_history > 0 else []
    for i, step in enumerate(history, start=1):
        hist_image_path = resolve_image_path(img_dir, step["img_filename"])
        if os.path.exists(hist_image_path):
            content.append({"type": "text", "text": f"History screenshot {i}:"})
            content.append(image_content(hist_image_path))
        thought = step.get("thoughts")
        answer = get_answer(step)
        if thought:
            content.append({"type": "text", "text": f"History action {i}: Thought: {thought}\nAction: {json.dumps(answer, ensure_ascii=False)}"})
        else:
            content.append({"type": "text", "text": f"History action {i}: {json.dumps(answer, ensure_ascii=False)}"})

    content.append({"type": "text", "text": "Current screenshot:"})
    content.append(image_content(current_image_path))
    return [{"role": "user", "content": content}], current_image_path


def extract_point(action_str):
    match = re.search(r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", action_str)
    if not match:
        match = re.search(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", action_str)
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))]


def extract_quoted_arg(action_str, names):
    for name in names:
        match = re.search(name + r"\s*=\s*(['\"])(.*?)\1", action_str, flags=re.DOTALL)
        if match:
            return match.group(2)
    return ""


def point_to_normalized(point, img_size):
    img_width, img_height = img_size
    h_resized, w_resized = smart_resize(img_height, img_width)
    x_original = point[0] * (img_width / w_resized)
    y_original = point[1] * (img_height / h_resized)
    return [round(x_original / img_width, 4), round(y_original / img_height, 4)]


def parse_json_action(action_str):
    try:
        action = json.loads(action_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict) or "action" not in action:
        return None
    return {
        "action": str(action["action"]).upper(),
        "value": action.get("value"),
        "position": action.get("position"),
    }


def parse_model_response(response_text, img_size):
    parts = response_text.split("Action:", 1)
    if len(parts) != 2:
        return None
    action_str = parts[1].strip()

    if action_str.startswith("{"):
        return parse_json_action(action_str)

    action_lower = action_str.lower()
    if action_lower.startswith("click") or "click(" in action_lower:
        point = extract_point(action_str)
        if point is None:
            return None
        return {"action": "CLICK", "value": None, "position": point_to_normalized(point, img_size)}
    if action_lower.startswith("type") or "type(" in action_lower or "input" in action_lower:
        return {
            "action": "TYPE",
            "value": extract_quoted_arg(action_str, ["content", "text", "value"]),
            "position": None,
        }
    if "scroll" in action_lower:
        direction = extract_quoted_arg(action_str, ["direction"])
        if not direction:
            for candidate in ["up", "down", "left", "right"]:
                if candidate in action_lower:
                    direction = candidate
                    break
        return {"action": f"SCROLL {direction.upper()}".strip(), "value": None, "position": None}
    if "press_back" in action_lower or "press back" in action_lower:
        return {"action": "PRESS BACK", "value": None, "position": None}
    if "press_home" in action_lower or "press home" in action_lower:
        return {"action": "PRESS HOME", "value": None, "position": None}
    if "press_enter" in action_lower or "press enter" in action_lower:
        return {"action": "PRESS ENTER", "value": None, "position": None}
    if "status_task_complete" in action_lower or "task complete" in action_lower:
        return {"action": "STATUS TASK COMPLETE", "value": None, "position": None}
    if "status_task_impossible" in action_lower or "task impossible" in action_lower:
        return {"action": "STATUS TASK IMPOSSIBLE", "value": None, "position": None}
    return None


def predict_action(client, item, img_dir, model, num_history, temperature, top_k, max_retries=3):
    messages, current_image_path = build_messages(item, img_dir, num_history)
    img_size = image_size(current_image_path)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
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
            )
            prediction_text = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            if usage is not None and hasattr(usage, "total_tokens"):
                print(f"Raw prediction: {prediction_text}; Usage: {usage.total_tokens}")
            else:
                print(f"Raw prediction: {prediction_text}")
            return parse_model_response(prediction_text, img_size), prediction_text
        except (OpenAIError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None, ""
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)


def is_tap_action(start_yx, end_yx):
    return np.linalg.norm(np.asarray(start_yx) - np.asarray(end_yx)) <= SWIPE_DISTANCE_THRESHOLD


def resize_annotation_bounding_boxes(annotation_positions):
    annotation_positions = np.asarray(annotation_positions, dtype=np.float32)
    height_change = ANNOTATION_HEIGHT_AUGMENT_FRACTION * annotation_positions[:, 2]
    width_change = ANNOTATION_WIDTH_AUGMENT_FRACTION * annotation_positions[:, 3]
    return np.stack(
        [
            np.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
            np.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
            np.minimum(1, annotation_positions[:, 2] + height_change),
            np.minimum(1, annotation_positions[:, 3] + width_change),
        ],
        axis=1,
    )


def yx_in_bounding_boxes(yx, boxes):
    y, x = yx
    top = boxes[:, 0]
    left = boxes[:, 1]
    bottom = top + boxes[:, 2]
    right = left + boxes[:, 3]
    return (y >= top) & (y <= bottom) & (x >= left) & (x <= right)


def check_tap_actions_match(tap_1_yx, tap_2_yx, annotation_positions):
    boxes = resize_annotation_bounding_boxes(annotation_positions)
    tap1_in_box = yx_in_bounding_boxes(tap_1_yx, boxes)
    tap2_in_box = yx_in_bounding_boxes(tap_2_yx, boxes)
    both_in_box = np.max(tap1_in_box & tap2_in_box) if len(boxes) else False
    within_threshold = np.linalg.norm(np.asarray(tap_1_yx) - np.asarray(tap_2_yx)) <= TAP_DISTANCE_THRESHOLD
    return bool(both_in_box or within_threshold)


def check_drag_actions_match(drag_1_touch_yx, drag_1_lift_yx, drag_2_touch_yx, drag_2_lift_yx):
    drag_1_deltas = np.asarray(drag_1_lift_yx) - np.asarray(drag_1_touch_yx)
    drag_2_deltas = np.asarray(drag_2_lift_yx) - np.asarray(drag_2_touch_yx)
    return bool(np.argmax(np.abs(drag_1_deltas)) == np.argmax(np.abs(drag_2_deltas)))


def check_actions_match(pred, ref, annotation_positions):
    pred_non_dual = pred["action_type"] != 4
    ref_non_dual = ref["action_type"] != 4
    if pred_non_dual or ref_non_dual:
        return pred["action_type"] == ref["action_type"]

    pred_is_tap = is_tap_action(pred["touch_point"], pred["lift_point"])
    ref_is_tap = is_tap_action(ref["touch_point"], ref["lift_point"])
    if pred_is_tap != ref_is_tap:
        return False
    if pred_is_tap and ref_is_tap:
        return check_tap_actions_match(pred["touch_point"], ref["touch_point"], annotation_positions)
    return check_drag_actions_match(pred["touch_point"], pred["lift_point"], ref["touch_point"], ref["lift_point"])


def action2json(step_data):
    action_type = step_data["action_type_id"]
    if action_type == 4:
        if step_data["action_type_text"] == "click":
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
        elif step_data["action_type_text"] == "scroll down":
            touch_point = [0.5, 0.8]
            lift_point = [0.5, 0.2]
        elif step_data["action_type_text"] == "scroll up":
            touch_point = [0.5, 0.2]
            lift_point = [0.5, 0.8]
        elif step_data["action_type_text"] == "scroll left":
            touch_point = [0.2, 0.5]
            lift_point = [0.8, 0.5]
        elif step_data["action_type_text"] == "scroll right":
            touch_point = [0.8, 0.5]
            lift_point = [0.2, 0.5]
        else:
            touch_point = [-1.0, -1.0]
            lift_point = [-1.0, -1.0]
    else:
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]

    typed_text = step_data["type_text"] if action_type == 3 else ""
    action = {"action_type": action_type, "touch_point": touch_point, "lift_point": lift_point, "typed_text": typed_text}
    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    if action["typed_text"] is not None:
        action["typed_text"] = action["typed_text"].lower()
    return action


def pred2json_post(step_data):
    action2id = {
        "CLICK": 4,
        "TYPE": 3,
        "SELECT": 2,
        "SCROLL UP": 1,
        "SCROLL DOWN": 0,
        "SCROLL LEFT": 8,
        "SCROLL RIGHT": 9,
        "SCROLL_UP": 1,
        "SCROLL_DOWN": 0,
        "SCROLL_LEFT": 8,
        "SCROLL_RIGHT": 9,
        "PRESS BACK": 5,
        "PRESS HOME": 6,
        "PRESS ENTER": 7,
        "STATUS TASK COMPLETE": 10,
        "STATUS TASK IMPOSSIBLE": 11,
        "PRESS_BACK": 5,
        "PRESS_HOME": 6,
        "PRESS_ENTER": 7,
        "STATUS_TASK_COMPLETE": 10,
        "STATUS_TASK_IMPOSSIBLE": 11,
    }
    action_type = str(step_data["action"]).upper()
    if action_type not in action2id:
        return None
    action_id = action2id[action_type]

    if action_id == 4:
        action_type_new = 4
        if step_data.get("position") is None:
            return None
        touch_point = step_data["position"]
        lift_point = step_data["position"]
        typed_text = ""
    elif action_id == 0:
        action_type_new = 4
        touch_point = [0.5, 0.8]
        lift_point = [0.5, 0.2]
        typed_text = ""
    elif action_id == 1:
        action_type_new = 4
        touch_point = [0.5, 0.2]
        lift_point = [0.5, 0.8]
        typed_text = ""
    elif action_id == 8:
        action_type_new = 4
        touch_point = [0.2, 0.5]
        lift_point = [0.8, 0.5]
        typed_text = ""
    elif action_id == 9:
        action_type_new = 4
        touch_point = [0.8, 0.5]
        lift_point = [0.2, 0.5]
        typed_text = ""
    else:
        action_type_new = action_id
        touch_point = [-1.0, -1.0]
        lift_point = [-1.0, -1.0]
        typed_text = step_data.get("value") if action_type_new == 3 else ""

    action = {"action_type": action_type_new, "touch_point": touch_point, "lift_point": lift_point, "typed_text": typed_text}
    action["touch_point"] = [action["touch_point"][1], action["touch_point"][0]]
    action["lift_point"] = [action["lift_point"][1], action["lift_point"][0]]
    if action["typed_text"] is not None:
        action["typed_text"] = action["typed_text"].lower()
    return action


def empty_step_result(item, prediction_text, action_pred):
    return {
        "domain": item.get("domain", "unknown"),
        "anno_id": item.get("anno_id"),
        "ep_id": item.get("ep_id", "unknown"),
        "img_path": item.get("img_url"),
        "instruction": item.get("task"),
        "sentence": prediction_text,
        "prediction": action_pred,
        "answer": get_answer(item),
        "corr_action": 0,
        "corr_type": 0,
        "num_text": 0,
        "corr_text": 0,
        "num_scroll": 0,
        "corr_scroll": 0,
        "num_click": 0,
        "corr_click": 0,
        "num_both_click": 0,
        "corr_both_click": 0,
        "num_wrong_format": 0,
    }


def evaluate_prediction(item, action_pred, prediction_text):
    step_result = empty_step_result(item, prediction_text, action_pred)
    try:
        action_pred_json = pred2json_post(action_pred)
        if action_pred_json is None:
            step_result["num_wrong_format"] += 1
            return step_result

        action_ref = action2json(item)
        annot_position = item.get("annot_position", [])
        annot_position = np.array([annot_position[i : i + 4] for i in range(0, len(annot_position), 4)], dtype=np.float32)
        match_label = check_actions_match(action_pred_json, action_ref, annot_position)

        if match_label:
            step_result["corr_action"] += 1
        if action_pred_json["action_type"] == action_ref["action_type"]:
            step_result["corr_type"] += 1
        if action_ref["action_type"] == 3:
            step_result["num_text"] += 1
            pred_text = action_pred_json["typed_text"] or ""
            ref_text = action_ref["typed_text"] or ""
            if pred_text == ref_text or pred_text in ref_text or ref_text in pred_text:
                step_result["corr_text"] += 1
        if action_ref["action_type"] == 4:
            if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                step_result["num_click"] += 1
                if match_label:
                    step_result["corr_click"] += 1
            else:
                step_result["num_scroll"] += 1
                if match_label:
                    step_result["corr_scroll"] += 1
            if (
                action_pred_json["action_type"] == 4
                and is_tap_action(action_ref["touch_point"], action_ref["lift_point"])
                and is_tap_action(action_pred_json["touch_point"], action_pred_json["lift_point"])
            ):
                step_result["num_both_click"] += 1
                if match_label:
                    step_result["corr_both_click"] += 1
    except Exception as e:
        print(f"Format Action met error: {e}")
        step_result["num_wrong_format"] += 1
    return step_result


def calculate_aitw_metrics(results):
    sums = defaultdict(int)
    num = 0
    for episode_steps in results.values():
        for step in episode_steps:
            for key in [
                "corr_action",
                "corr_type",
                "num_text",
                "corr_text",
                "num_scroll",
                "corr_scroll",
                "num_click",
                "corr_click",
                "num_both_click",
                "corr_both_click",
                "num_wrong_format",
            ]:
                sums[key] += step[key]
            num += 1
    if num == 0:
        return {"Score": 0.0, "Num": 0}
    return {
        "Score": sums["corr_action"] / num,
        "Num Corr Action": sums["corr_action"],
        "Num Corr Type": sums["corr_type"],
        "Num Text": sums["num_text"],
        "Num Corr Text": sums["corr_text"],
        "Num Scroll": sums["num_scroll"],
        "Num Corr Scroll": sums["corr_scroll"],
        "Num Click": sums["num_click"],
        "Num Corr Click": sums["corr_click"],
        "Num Both Click": sums["num_both_click"],
        "Num Corr Both Click": sums["corr_both_click"],
        "Num Wrong Format": sums["num_wrong_format"],
        "Num": num,
    }


def evaluate_item(idx, item, img_dir, client, model, num_history, temperature, top_k):
    item = dict(item)
    item["anno_id"] = idx
    current_image_path = resolve_image_path(img_dir, item["img_url"])
    if not os.path.exists(current_image_path):
        print(f"Image not found: {current_image_path}")
        return None
    action_pred, prediction_text = predict_action(client, item, img_dir, model, num_history, temperature, top_k)
    if action_pred is None:
        result = empty_step_result(item, prediction_text, None)
        result["num_wrong_format"] += 1
        return result
    return evaluate_prediction(item, action_pred, prediction_text)


def print_metrics(metrics_by_domain):
    print("\n===== AITW METRICS =====")
    scores = []
    for domain, metrics in metrics_by_domain.items():
        score = metrics.get("Score", 0.0)
        scores.append(score)
        print(f"{domain}: {score:.4f} ({metrics.get('Num Corr Action', 0)}/{metrics.get('Num', 0)})")
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Average Score: {avg_score:.4f}")


def main():
    model = os.environ.get("MODEL", "dart-7b")
    endpoint = os.environ.get("ENDPOINT", "http://localhost:8000/v1")
    dataset_dir = os.environ.get("AITW_DATASET_DIR", "dataset")
    version = os.environ.get("AITW_VERSION", "v2")
    limit = int(os.environ.get("LIMIT", "-1"))
    num_history = int(os.environ.get("NUM_HISTORY", "2"))
    max_workers = int(os.environ.get("AITW_MAX_WORKERS", "8"))
    temperature = float(os.environ.get("TEMPERATURE", "1e-6"))
    top_k = int(os.environ.get("TOP_K", "50"))

    client = openai.OpenAI(api_key="EMPTY", base_url=endpoint)
    json_data, img_dir = load_aitw_data(dataset_dir, version=version, limit=limit)

    results_by_domain = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_item, idx, item, img_dir, client, model, num_history, temperature, top_k)
            for idx, item in enumerate(json_data)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as e:
                print(f"Skipping sample because evaluation failed: {e}")
                continue
            if result is None:
                continue
            domain = result["domain"]
            ep_id = result["ep_id"]
            results_by_domain[domain][ep_id].append(result)

    metrics_by_domain = {}
    for domain, domain_results in results_by_domain.items():
        metrics_by_domain[domain] = calculate_aitw_metrics(domain_results)

    avg_score = (
        sum(metrics["Score"] for metrics in metrics_by_domain.values()) / len(metrics_by_domain)
        if metrics_by_domain
        else 0.0
    )
    output = {
        "metrics": metrics_by_domain,
        "avg_score": avg_score,
        "results": results_by_domain,
    }
    print_metrics(metrics_by_domain)
    with open(f"results_{model}_aitw.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
