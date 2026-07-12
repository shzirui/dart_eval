import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import openai
from PIL import Image, ImageDraw
from tqdm import tqdm
from io import BytesIO


import math
import re

def round_by_factor(x, factor): return int(round(x / factor) * factor)
def floor_by_factor(x, factor): return int(math.floor(x / factor) * factor)
def ceil_by_factor(x, factor): return int(math.ceil(x / factor) * factor)

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

def extract_point(response):
    parts = response.split("Action:", 1)
    if len(parts) != 2:
        return None
    action_part = parts[1]
    match = re.search(r"<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>", action_part)
    if not match:
        match = re.search(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", action_part)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None

def load_screenspot_data(
        dataset_dir,
        limit=-1, 
        filter_by_split: str | None = None
    ):
    # Recreating the dataset loading logic without using the class
    base_image_dir = os.path.join(dataset_dir, "ScreenSpot-v2")
    meta_dir = base_image_dir
    img_dir = os.path.join(base_image_dir, "screenspotv2_image")

    # Load the JSON data
    json_data = []
    for filename in os.listdir(meta_dir):
        if filename.endswith(".json"):
            with open(os.path.join(meta_dir, filename)) as f:
                file_data = json.load(f)
                for item in file_data:
                    # Add the filename to the item
                    item['filename'] = filename.removeprefix("screenspot_").removesuffix("_v2.json")
                    json_data.append(item)

    print(f"Dataset: Screenspot; # samples: {len(json_data)}")
    if limit > 0:
        json_data = json_data[:limit]
    # if filter_by_split:
    #     json_data = [item for item in json_data if item['split'] == filter_by_split]
    return json_data, img_dir

def predict_point(client: openai.OpenAI, image_path: str, task: str, model: str = "tongui-3b"):
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
 
    buffered = BytesIO()
    with Image.open(image_path).convert("RGB") as img:
        # img = resize_image(
        #     img,
        #     max_pixels=2500*28*28
        # )
        img_width, img_height = img.size
        # img.save(buffered, format="JPEG")
        # encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": 
                [
                    {"type": "text", "text": f"You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>')\n\n## User Instruction {task}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            },
        ],
        max_completion_tokens=500,
        temperature=1e-6,
        top_p=0.95,
        extra_body={
            "top_k": 50,
            "mm_processor_kwargs":{
                "min_pixels":1000 *28 *28,
                "max_pixels":16384 *28 *28
            }
        }
    )
    
    # Extract the point from the response
    prediction_text = response.choices[0].message.content
    print(f"Raw prediction: {prediction_text}; Usage: {response.usage.total_tokens}")
    
    # Try to parse the point from the response
    try:
        # The model might return a point as [x, y] or a bounding box as [x1, y1, x2, y2]
        coords = extract_point(prediction_text)

        if not coords:
            return None

        h_resized, w_resized = smart_resize(img_height, img_width)
        
        if len(coords) == 2:  # It's a point [x, y]
            x = int(coords[0] * (img_width / w_resized))
            y = int(coords[1] * (img_height / h_resized))
            return [x, y]
        else:
            print(f"Unexpected format: {coords}")
            return None
            
    except json.JSONDecodeError:
        print(f"Error parsing coordinates: {prediction_text}")
        return None

def is_point_in_box(point, box):
    """
    Check if a point [x, y] is inside a bounding box [x1, y1, x2, y2]
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def display_point_on_image(image_path, point, gt_box=None, save_path=None):
    """
    Display a point on an image, optionally with a ground truth box
    point format: [x, y]
    gt_box format: [x1, y1, x2, y2]
    """
    import matplotlib.pyplot as plt
    import numpy as np

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw the point (as a small circle)
    point_radius = 5
    x, y = point
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], 
        fill="red", outline="red"
    )
    
    # Draw the ground truth box if provided
    if gt_box:
        draw.rectangle([(gt_box[0], gt_box[1]), (gt_box[2], gt_box[3])], outline="green", width=2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(img))  # Convert PIL Image to numpy array
    plt.axis('off')
    
    # Add a title indicating if the point is in the box
    if gt_box:
        is_in_box = is_point_in_box(point, gt_box)
        plt.title(f"Point in box: {is_in_box}", fontsize=14)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def print_accuracy_metrics(group_metrics):
    """
    Print accuracy metrics for different groups
    """
    print("\n===== ACCURACY METRICS =====")
    
    # 1. Overall accuracy
    total_correct = sum(metrics['correct'] for metrics in group_metrics.values())
    total_samples = sum(metrics['total'] for metrics in group_metrics.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"\n1. Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    
    # 2. Accuracy by split and data_type
    print("\n2. Accuracy by Split and Data Type:")
    
    # Group by split
    split_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for (split, data_type), metrics in group_metrics.items():
        split_metrics[split]['correct'] += metrics['correct']
        split_metrics[split]['total'] += metrics['total']
    
    print("  By Split:")
    for split, metrics in split_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {split}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Group by data_type
    data_type_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for (split, data_type), metrics in group_metrics.items():
        data_type_metrics[data_type]['correct'] += metrics['correct']
        data_type_metrics[data_type]['total'] += metrics['total']
    
    print("  By Data Type:")
    for data_type, metrics in data_type_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {data_type}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Detailed breakdown by split and data_type
    print("  By Split and Data Type:")
    for (split, data_type), metrics in group_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {split}, {data_type}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # 3. Average accuracy across groups
    print("\n3. Average Accuracy Across Groups:")
    
    # Average by split
    split_avg = {}
    for split, metrics in split_metrics.items():
        split_avg[split] = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    
    avg_split_accuracy = sum(split_avg.values()) / len(split_avg) if split_avg else 0
    print(f"  Average across splits: {avg_split_accuracy:.2f}%")
    
    # Average by data_type
    data_type_avg = {}
    for data_type, metrics in data_type_metrics.items():
        data_type_avg[data_type] = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    
    avg_data_type_accuracy = sum(data_type_avg.values()) / len(data_type_avg) if data_type_avg else 0
    print(f"  Average across data types: {avg_data_type_accuracy:.2f}%")
    
    # Average across all groups
    group_accuracies = []
    for (split, data_type), metrics in group_metrics.items():
        if metrics['total'] > 0:
            group_accuracies.append(metrics['correct'] / metrics['total'] * 100)
    
    avg_group_accuracy = sum(group_accuracies) / len(group_accuracies) if group_accuracies else 0
    print(f"  Average across all groups: {avg_group_accuracy:.2f}%")

def main():
    # Set to True to display points on images
    DISPLAY_IMAGES = False
    LIMIT = -1
    MAX_WORKERS = int(os.environ.get("SCREENSPOT_MAX_WORKERS", "8"))
    # MODEL = "tongui-3b"
    # MODEL = "dart-7b"
    # MODEL = "ui-tars-2b"
    # MODEL = "ui-tars-7b"
    # MODEL = "qwen2.5-7b"
    MODEL = "dart-7b"
    ENDPOINT = "http://hgx-hyperplane04:8000/v1"
    FILTER_BY_SPLIT = None
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=ENDPOINT,
    )
    # Load the dataset
    json_data, img_dir = load_screenspot_data(
        "dataset",
        limit=LIMIT,
        filter_by_split=FILTER_BY_SPLIT
    )

    # Track metrics by group (split and data_type)
    group_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})

    def evaluate_item(idx, item):
        image_path = os.path.join(img_dir, item['img_filename'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        task = item['instruction']
        gt_bbox = item['bbox']  # [x, y, width, height]
        gt_bbox_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
        print(f"Ground truth bbox [x1, y1, x2, y2]: {gt_bbox_xyxy}")

        pred_point = predict_point(client, image_path, task, model=MODEL)
        print(f"Predicted point [x, y]: {pred_point}")

        if pred_point is None:
            print("Skipping sample because prediction could not be parsed")
            return None

        is_correct = is_point_in_box(pred_point, gt_bbox_xyxy)
        print(f"Point in box: {is_correct}")

        if DISPLAY_IMAGES:
            output_dir = "point_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"sample_{idx}_{os.path.basename(item['img_filename'])}")
            display_point_on_image(image_path, pred_point, gt_bbox_xyxy, save_path)

        split = item.get('filename', 'unknown')
        data_type = item.get('data_type', 'unknown')
        return (split, data_type), is_correct

    # Process samples in parallel. vLLM batches concurrent requests internally.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(evaluate_item, idx, item) for idx, item in enumerate(json_data)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
            except Exception as e:
                print(f"Skipping sample because evaluation failed: {e}")
                continue
            if result is None:
                continue
            group_key, is_correct = result
            group_metrics[group_key]['total'] += 1
            if is_correct:
                group_metrics[group_key]['correct'] += 1
    
    # Print accuracy metrics
    print_accuracy_metrics(group_metrics)
    serializable_metrics = {
        f"{split}/{data_type}": metrics
        for (split, data_type), metrics in group_metrics.items()
    }
    with open(f"results_{MODEL}_screenspot_2.json", "w") as f:
        json.dump(serializable_metrics, f, indent=4)

if __name__ == "__main__":
    main()
