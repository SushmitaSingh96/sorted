import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# 1. Load CLIP model + processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. Load image dataset
dataset_path = "microscopic_images"
image_paths = []
labels = []
label_map = {}  # string -> int
label_counter = 0

for img_file in os.listdir(dataset_path):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        allowed_labels = ["cotton", "viscose", "polyester"]
        label_str = ''.join([c for c in img_file if not c.isdigit()]).replace('.JPG', '').replace('.jpeg', '').replace('.png', '').lower()
        if label_str not in allowed_labels:
            continue  # skip unknown fabrics
        if label_str not in label_map:
            label_map[label_str] = label_counter
            label_counter += 1
        image_paths.append(os.path.join(dataset_path, img_file))
        labels.append(label_map[label_str])

# 3. Extract features
def extract_features(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()

print("Extracting features...")
features = np.array([extract_features(p) for p in image_paths])
labels = np.array(labels)

# Optional: Save features for later reuse
np.save("clip_features.npy", features)
np.save("clip_labels.npy", labels)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

# 5. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Save the trained classifier
joblib.dump(clf, "clip_logistic_classifier.pkl")
joblib.dump(label_map, "label_map.pkl")

print("Training complete. Model saved as 'clip_logistic_classifier.pkl'")