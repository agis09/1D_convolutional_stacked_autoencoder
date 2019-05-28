from model import DeepAutoEncoder, \
    AutoEncoderStack01, AutoEncoderStack02, AutoEncoderStack03, \
    AutoEncoderStack04

import numpy as np
import sys
import pandas as pd


def main():
    data_path = "../mouse_cell_classification/N_lab_2019_03_mouse_cell_classification/data/data.csv"
    df = pd.read_csv(data_path)
    x_train1 = df.loc[:, "591":"1719"].values
    x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
    x_test1 = x_train1
    print(x_train1.shape)

    # step1
    print("***** STEP 1 *****")
    ae01 = AutoEncoderStack01(x_train1)
    ae01.compile()
    ae01.train(x_train=x_train1, x_test=x_test1, nb_epoch=100, batch_size=128)

    enc_train1 = ae01.encoder.predict(x=x_train1)
    enc_test1 = ae01.encoder.predict(x=x_test1)

    np.save('train_stack01.npy', enc_train1)
    np.save('test_stack01.npy', enc_test1)

    del enc_train1, enc_test1

    # step2
    print("***** STEP 2 *****")

    x_train2 = np.load('./save_data/train_stack01.npy')
    x_test2 = np.load('./save_data/test_stack01.npy')
    ae02 = AutoEncoderStack02(x_train2)
    ae02.compile()
    ae02.train(x_train=x_train2, x_test=x_test2, nb_epoch=100, batch_size=128)

    enc_train2 = ae02.encoder.predict(x=x_train2)
    enc_test2 = ae02.encoder.predict(x=x_test2)

    np.save('./save_data/train_stack02.npy', enc_train2)
    np.save('./save_data/test_stack02.npy', enc_test2)

    del x_train2, x_test2, enc_train2, enc_test2

    # step3
    print("***** STEP 3 *****")

    x_train3 = np.load('./save_data/train_stack02.npy')
    x_test3 = np.load('./save_data/test_stack02.npy')
    ae03 = AutoEncoderStack03(x_train3)
    ae03.compile()
    ae03.train(x_train=x_train3, x_test=x_test3, nb_epoch=100, batch_size=128)

    enc_train3 = ae03.encoder.predict(x=x_train3)
    enc_test3 = ae03.encoder.predict(x=x_test3)

    np.save('./save_data/train_stack03.npy', enc_train3)
    np.save('./save_data/test_stack03.npy', enc_test3)

    del x_train3, x_test3, enc_train3, enc_test3

    # step4
    print("***** STEP 4 *****")
    x_train4 = np.load('./save_data/train_stack03.npy')
    x_test4 = np.load('./save_data/test_stack03.npy')
    ae04 = AutoEncoderStack04(x_train4)
    ae04.compile()
    ae04.train(x_train=x_train4, x_test=x_test4, nb_epoch=100, batch_size=128)

    enc_train4 = ae04.encoder.predict(x=x_train4)
    enc_test4 = ae04.encoder.predict(x=x_test4)

    np.save('./save_data/train_stack04.npy', enc_train4)
    np.save('./save_data/test_stack04.npy', enc_test4)

    del x_train4, x_test4, enc_train4, enc_test4

    # step5
    print("***** STEP 5 *****")
    stacked_ae = DeepAutoEncoder(x_train1)
    # stacked_ae.load_weights(ae01=ae01.autoencoder, ae02=ae02.autoencoder, ae03=ae03.autoencoder, ae04=ae04.autoencoder)
    stacked_ae.compile()
    stacked_ae.train(x_train=x_train1, x_test=x_test1,
                     nb_epoch=100, batch_size=128)


if __name__ == "__main__":
    main()
