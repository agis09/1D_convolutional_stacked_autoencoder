import numpy as np
from model import DeepAutoEncoder
import cv2
import os
from keras.models import load_model

ae01 = load_model('./save_data/stack01_autoencoder.h5')
ae02 = load_model('./save_data/stack02_autoencoder.h5')
ae03 = load_model('./save_data/stack03_autoencoder.h5')
ae04 = load_model('./save_data/stack04_autoencoder.h5')

stacked_ae = DeepAutoEncoder()
stacked_ae.load_weights(ae01, ae02, ae03, ae04)

data_dir_path = "./animeface-character-dataset/thumb/"
tmp = os.listdir(data_dir_path)
dmp = sorted([x for x in tmp if os.path.isdir(data_dir_path)])
dir_list = tmp
for dir_name in dir_list:
    file_list = os.listdir(data_dir_path + dir_name)
    print(dir_name)
    for file_name in file_list:
        if file_name.endswith('.png'):
            img_path = str(data_dir_path) + str(dir_name) + '/' + str(file_name)
            img = cv2.imread(img_path)
            img = img.astype('float32') / 255.0
            img = cv2.resize(img, (80, 80))
            img = np.array(img)
            img = img.reshape(1, 80, 80, 3)
            cv2.imwrite("./result/" + str(file_name) + "_original.bmp", img[0] * 255.0)
            recon_img = stacked_ae.autoencoder.predict(x=img)
            cv2.imwrite("./result/" + str(file_name) + "_feture.bmp", recon_img[0] * 255.0)
    break
