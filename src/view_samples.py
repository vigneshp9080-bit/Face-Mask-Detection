import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "processed_dataset"
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

plt.figure(figsize=(10, 6))

img_no = 1

for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    image_name = os.listdir(class_path)[0]
    image_path = os.path.join(class_path, image_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 3, img_no)
    plt.imshow(image)
    plt.title(cls)
    plt.axis("off")

    img_no += 1

plt.show()