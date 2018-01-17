import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

data_dir_path = "./animeface-character-dataset/thumb/"
tmp = os.listdir(data_dir_path)
dmp = sorted([x for x in tmp if os.path.isdir(data_dir_path)])
dir_list = tmp
# print(dir_list)

X_target = []
for dir_name in dir_list:
    file_list = os.listdir(data_dir_path + dir_name)
    print(dir_name)
    for file_name in file_list:
        if file_name.endswith('.png'):
            image_path = str(data_dir_path) + str(dir_name) + '/' + str(file_name)
            image = cv.imread(image_path)
            image = cv.resize(image, (80, 80))
            X_target.append(image)
    break
X_target = np.array(X_target)
# print(X_target.shape)
X_train, X_test = train_test_split(X_target, test_size=0.2)
np.save('train.npy', X_train)
np.save('test.npy', X_test)
print(X_target.shape)   # (データ数、高さ、幅、チャンネル数)