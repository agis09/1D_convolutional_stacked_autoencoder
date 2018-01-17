from model import DeepAutoEncoder, \
    AutoEncoderStack01, AutoEncoderStack02, AutoEncoderStack03, \
    AutoEncoderStack04

import numpy as np
import cv2
import sys


def main():
    # (データ数、高さ、幅、チャンネル数)
    x_train1 = np.load('train.npy')
    x_test1 = np.load('test.npy')

    # データ拡張
    argumented_xs = list()
    for i in range(x_train1.shape[0]):
        for k in range(5):
            if k == 0:
                x = x_train1[i, 20:, 20:, :]
            elif k == 1:
                x = x_train1[i, :-20, 20:, :]
            elif k == 2:
                x = x_train1[i, :-20, :-20, :]
            elif k == 3:
                x = x_train1[i, 20:, :-20, :]
            else:
                x = x_train1[i, 10:-10, 10:-10, :]

            x2 = cv2.resize(x, (80, 80))
            argumented_xs.append(x2)

    x_train1_2 = np.concatenate((x_train1, np.array(argumented_xs)), axis=0)
    x_train1_2 = x_train1_2.astype(np.float32) / 255.0
    x_train1_2[x_train1_2 < 0.0] = 0.0
    x_train1_2[x_train1_2 > 1.0] = 1.0

    x_test1 = x_test1.astype(np.float32) / 255.0
    x_test1[x_test1 < 0.0] = 0.0
    x_test1[x_test1 > 1.0] = 1.0

    del x_train1

    # step1
    print("***** STEP 1 *****")
    ae01 = AutoEncoderStack01()
    ae01.compile()
    ae01.train(x_train=x_train1_2, x_test=x_test1, nb_epoch=100, batch_size=128)

    enc_train1 = ae01.encoder.predict(x=x_train1_2)
    enc_test1 = ae01.encoder.predict(x=x_test1)

    np.save('train_stack01.npy', enc_train1)
    np.save('test_stack01.npy', enc_test1)

    del enc_train1, enc_test1

    # step2
    print("***** STEP 2 *****")
    ae02 = AutoEncoderStack02()
    ae02.compile()
    x_train2 = np.load('./save_data/train_stack01.npy')
    x_test2 = np.load('./save_data/test_stack01.npy')
    ae02.train(x_train=x_train2, x_test=x_test2, nb_epoch=100, batch_size=128)

    enc_train2 = ae02.encoder.predict(x=x_train2)
    enc_test2 = ae02.encoder.predict(x=x_test2)

    np.save('train_stack02.npy', enc_train2)
    np.save('test_stack02.npy', enc_test2)

    del x_train2, x_test2, enc_train2, enc_test2

    # step3
    print("***** STEP 3 *****")
    ae03 = AutoEncoderStack03()
    ae03.compile()
    x_train3 = np.load('./save_data/train_stack02.npy')
    x_test3 = np.load('./save_data/test_stack02.npy')
    ae03.train(x_train=x_train3, x_test=x_test3, nb_epoch=100, batch_size=128)

    enc_train3 = ae03.encoder.predict(x=x_train3)
    enc_test3 = ae03.encoder.predict(x=x_test3)

    np.save('./save_data/train_stack03.npy', enc_train3)
    np.save('./save_data/test_stack03.npy', enc_test3)

    del x_train3, x_test3, enc_train3, enc_test3

    # step4
    print("***** STEP 4 *****")
    ae04 = AutoEncoderStack04()
    ae04.compile()
    x_train4 = np.load('./save_data/train_stack03.npy')
    x_test4 = np.load('./save_data/test_stack03.npy')
    ae04.train(x_train=x_train4, x_test=x_test4, nb_epoch=100, batch_size=128)

    enc_train4 = ae04.encoder.predict(x=x_train4)
    enc_test4 = ae04.encoder.predict(x=x_test4)

    np.save('./save_data/train_stack04.npy', enc_train4)
    np.save('./save_data/test_stack04.npy', enc_test4)

    del x_train4, x_test4, enc_train4, enc_test4

    # step5
    print("***** STEP 5 *****")
    stacked_ae = DeepAutoEncoder()
    # stacked_ae.load_weights(ae01=ae01.autoencoder, ae02=ae02.autoencoder, ae03=ae03.autoencoder, ae04=ae04.autoencoder)
    stacked_ae.compile()
    stacked_ae.train(x_train=x_train1_2, x_test=x_test1, nb_epoch=100, batch_size=128)


if __name__ == "__main__":
    main()