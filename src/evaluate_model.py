import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

IMG_SIZE = 128
BATCH_SIZE = 32

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "processed_dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_dataset.class_names

model = tf.keras.models.load_model("model/mask_model.h5")

y_true = np.concatenate([y for x, y in val_dataset], axis=0)

predictions = model.predict(val_dataset)

y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

print("\nConfusion Matrix\n")
print(confusion_matrix(y_true, y_pred))

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)