import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = "model/mobilenet_mask_model.keras"
CLASS_PATH = "model/class_names.json"

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "processed_dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
for images, labels in val_dataset:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

predictions = model.predict(val_dataset)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")
print(cm)

os.makedirs("screenshots", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Face Mask Detection")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("screenshots/confusion_matrix.png", dpi=300)
plt.show()

print("Confusion matrix saved at screenshots/confusion_matrix.png")