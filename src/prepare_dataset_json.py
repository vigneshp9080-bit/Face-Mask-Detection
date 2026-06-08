import os
import json
import cv2

# Dataset paths
images_path = "dataset/img"
annotations_path = "dataset/ann"
output_path = "processed_dataset"

# Classes
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

# Create output class folders
for cls in classes:
    os.makedirs(os.path.join(output_path, cls), exist_ok=True)

count = 0

# Loop all annotation JSON files
for ann_file in os.listdir(annotations_path):

    if not ann_file.endswith(".json"):
        continue

    ann_path = os.path.join(annotations_path, ann_file)

    # Image name example:
    # maksssksksss0.png.json -> maksssksksss0.png
    image_name = ann_file.replace(".json", "")
    image_path = os.path.join(images_path, image_name)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found:", image_path)
        continue

    # Read JSON annotation
    with open(ann_path, "r") as file:
        data = json.load(file)

    # Each object means one face
    for obj in data["objects"]:

        label = obj["classTitle"]

        if label not in classes:
            continue

        # Get bounding box points
        points = obj["points"]["exterior"]

        x1, y1 = points[0]
        x2, y2 = points[1]

        # Crop face from image
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Save cropped face
        save_name = f"{label}_{count}.jpg"
        save_path = os.path.join(output_path, label, save_name)

        cv2.imwrite(save_path, face)

        count += 1

print("Dataset preparation completed")
print("Total cropped faces:", count)