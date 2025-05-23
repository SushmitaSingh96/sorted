import sys
import torch
import numpy as np
from PIL import Image
import joblib
from transformers import CLIPProcessor, CLIPModel
import os
import matplotlib.pyplot as plt

# 1. Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. Load trained classifier and label map
clf = joblib.load("clip_logistic_classifier.pkl")
label_map = joblib.load("label_map.pkl")
inv_label_map = {v: k for k, v in label_map.items()}  # int -> string

# 3. Load and process image
def extract_features(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()

import json
import subprocess

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    features = extract_features(image_path).reshape(1, -1)
    prediction = clf.predict(features)[0]
    fabric_type = inv_label_map[prediction]

    # Fabric-to-score mapping
    sustainability_scores = {
        "cotton": 6,
        "polyester": 3,
        "viscose": 4
    }
    sustainability_score = sustainability_scores.get(fabric_type.lower(), 0)

    print(f"Predicted fabric: {fabric_type}")

    # Save result to JSON
    dashboard_data = {
        "image_path": image_path,
        "fabric": fabric_type,
        "score": sustainability_score
    }

    with open("dashboard_input.json", "w") as f:
        json.dump(dashboard_data, f)

    # Launch dashboard
    subprocess.Popen(["streamlit", "run", "dashboard.py"])

# 5. Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    predict_image(image_path)