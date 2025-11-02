# ==============================
# Task 03 - Cats vs Dogs with SVM (Fast Version)
# ==============================

# Step 1: Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Define dataset path
# Download the Kaggle dataset and extract it:
# PetImages/Cat/*.jpg and PetImages/Dog/*.jpg
DATASET_DIR = r"cat_dog_dataset_task3\PetImages"  # Use raw string for Windows path

# Step 3: Prepare data
IMG_SIZE = 32      # Smaller size = faster training
MAX_IMAGES = 500   # Max images per class for faster training
data = []
labels = []

# Loop through both classes
for category in ["Cat", "Dog"]:
    folder_path = os.path.join(DATASET_DIR, category)
    label = 0 if category == "Cat" else 1
    count = 0

    for file in os.listdir(folder_path):
        if count >= MAX_IMAGES:
            break
        # Only read valid image files
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
        if img is None:  # skip corrupted images
            continue

        # Resize image and flatten
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img.flatten())
        labels.append(label)
        count += 1

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)
print("✅ Dataset prepared successfully!")
print("Dataset shape:", X.shape, y.shape)  # e.g., (1000, 1024) (1000,)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train SVM model
print("🧠 Training SVM model... (should finish in <1 min)")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = svm_model.predict(X_test)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize a few predictions
def show_image(img_array, label, pred):
    plt.imshow(img_array.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"True: {'Cat' if label==0 else 'Dog'} | Pred: {'Cat' if pred==0 else 'Dog'}")
    plt.axis("off")
    plt.show()

for i in range(5):
    show_image(X_test[i], y_test[i], y_pred[i])
