import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 1. Load CLIP model + processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. Load your image dataset
# Assumes structure: dataset/class_name/image.jpg
# 2. Load your image dataset
# Assumes structure: microscopic_images/viscose3.jpg, cotton1.jpg, etc.
dataset_path = "microscopic_images"
image_paths = []
labels = []

label_map = {}  # string -> int
label_counter = 0

for img_file in os.listdir(dataset_path):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        label_str = ''.join([c for c in img_file if not c.isdigit()]).replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        if label_str not in label_map:
            label_map[label_str] = label_counter
            label_counter += 1
        image_paths.append(os.path.join(dataset_path, img_file))
        labels.append(label_map[label_str])

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue
    if class_name not in label_map:
        label_map[class_name] = label_counter
        label_counter += 1
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(class_folder, img_file))
            labels.append(label_map[class_name])

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

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

# 5. Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# 7. Optional: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()