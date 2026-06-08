import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "processed_dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "processed_dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

print("Training Dataset Loaded")
print("Classes:", train_dataset.class_names)

for images, labels in train_dataset.take(1):
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)
