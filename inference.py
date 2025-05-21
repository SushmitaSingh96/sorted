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

# 4. Predict and display image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    features = extract_features(image_path).reshape(1, -1)
    prediction = clf.predict(features)[0]
    fabric_type = inv_label_map[prediction]
    
    print(f"Predicted fabric: {fabric_type}")
    
    # Display image with predicted label
    plt.imshow(image)
    plt.title(f"Predicted Fabric: {fabric_type}", fontsize=16)
    plt.axis("off")
    plt.show()

# 5. CLI usage
if __name__ == "__main__":
    image_path = os.path.join("test_p.jpg")  # or use sys.argv[1]
    predict_image(image_path)