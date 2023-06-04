import numpy as np
import os
import cv2


dir = "C:\\Users\\Haadin\\PycharmProjects\\DIP_Project\\Queensland Dataset CE42\\Testing\\Images"

images = np.zeros([300, 256, 256, 3], dtype=np.uint8)
img_count = 0

print(images.shape)

labels = np.zeros([300, 3], dtype=np.uint8)
a = os.listdir(dir)
print(a)

for i in range(len(a)):
    b = os.listdir(os.path.join(dir, a[i]))
    print(len(b))
    for j in range(len(b)):
        img = cv2.imread(os.path.join(dir, a[i], b[j]), 1)
        labels[img_count, i] = 1    # One hot encoding
        images[img_count,...] = img # image insertion into numpy array
        img_count += 1

np.save("DIP_Project_img.npy", images)
np.save("DIP_Project_labels.npy", labels)

print("LABEL : ", labels[50,:])
cv2.imshow("test", images[50,...])

cv2.waitKey(0)


