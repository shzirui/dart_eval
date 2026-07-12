import json
from openai import OpenAI
import sys
from PIL import Image
import base64
sys.path.append(".")
import time
import os

client = OpenAI(
    base_url="https://sv-30018610-e39b-4050-a508-a4119570857f-8001-x-defau-bf20391a29.sproxy.hd-01.alayanew.com:22443/v1",
    api_key="empty",
)
with open("/share/ml/shangzirui/TongUI/dataset/baidu_jingyan_splits/baidu_jingyan_test_annotation_good.json", "r") as f:
    data = json.load(f)

def get_action(msg: str) -> dict:
    action_str = msg.split("Action:")[1].strip()
    return json.loads(action_str)

action_correct = 0
location_correct = 0
acc = 0
total = 0
for item in data:
    flag1 = 0
    flag2 = 0
    try:
        images = item["images"]
        
        prompt = item["messages"][0]["content"]
        
        image_tag = "<image>"
        text_msgs = prompt.split(image_tag)
        # print(item)
        messages = [
            {"role": "user", "content": []}
        ]
        # annotation box x1, y1, x2, y2
        points = item["points"]
        image_path = images[-1]
        screenshot = Image.open(os.path.join("/scratch/zhangbofei/Projects/Multimodal-CL/Multimodal-Agent-Tuning/AgentNet/AgentNet", image_path))
        width, height = screenshot.size
        x1, y1, x2, y2 = points
        x1 = x1 / width
        y1 = y1 / height
        x2 = x2 / width
        y2 = y2 / height
        
        for text_id, text_msg in enumerate(text_msgs):
            if len(text_msg) == 0:
                continue
            messages[0]["content"].append({"type": "text", "text": text_msg})
            image_path = images[text_id]
            
            with open(os.path.join("/scratch/zhangbofei/Projects/Multimodal-CL/Multimodal-Agent-Tuning/AgentNet/AgentNet", image_path), "rb") as image_file:
                # Read the file directly into base64
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})

        response = client.chat.completions.create(
            model="tongui-32b",
            messages=messages,
            max_tokens=128,
            temperature=0.0,
        )
        action_str = response.choices[0].message.content
        action = get_action(action_str)
        annotation = get_action(item["messages"][-1]["content"])
        print("Prediction: ", action_str)
        print("Annotation: ", annotation)
        if action["action"] == annotation["action"]:
            action_correct += 1
            flag1 = 1
            print("Action correct!")
        if action["position"] is not None:
            x, y = action["position"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                location_correct += 1
                flag2 = 1
                print("Location correct!")
            else:
                print("Location incorrect!")
        else:
            location_correct += 1
            flag2 = 1
            print("Location correct!")

        if flag1 == 1 and flag2 == 1:
            acc += 1
            print("Both action and location correct!")
            
        total += 1
    except Exception as e:
        print("Error: ", e)
        print("Invalid sample met")
        total += 1

print("Total Valid Samples: ", total)
print("Action correct: ", action_correct / total)
print("Location correct: ", location_correct / total)
print("acc: ", acc / total)