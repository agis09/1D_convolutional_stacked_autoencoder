from keras.layers import Input, MaxPooling1D, UpSampling1D, Convolution1D, Cropping1D
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers


class DeepAutoEncoder(object):
    def __init__(self, data):
        input_img = Input(shape=(data.shape[1], data.shape[2]))  # 0
        conv1 = Convolution1D(16, 7, activation='relu', border_mode='same', init='glorot_normal',
                              W_regularizer=regularizers.l2(0.0005))(input_img)  # 1
        pool1 = MaxPooling1D(2, border_mode='same')(conv1)  # 2

        conv2 = Convolution1D(32, 5, activation='relu', border_mode='same', init='glorot_normal',
                              W_regularizer=regularizers.l2(0.0005))(pool1)  # 3
        pool2 = MaxPooling1D(2, border_mode='same')(conv2)  # 4

        conv3 = Convolution1D(64, 3, activation='relu', border_mode='same', init='glorot_normal',
                              W_regularizer=regularizers.l2(0.0005))(pool2)  # 5
        pool3 = MaxPooling1D(2, border_mode='same')(conv3)  # 6

        conv4 = Convolution1D(128, 3, activation='relu', border_mode='same', init='glorot_normal',
                              W_regularizer=regularizers.l2(0.0005))(pool3)  # 7
        pool4 = MaxPooling1D(2, border_mode='same')(conv4)  # 8

        encoded = pool4

        unpool4 = UpSampling1D(2)(pool4)  # 9
        deconv3 = Convolution1D(64, 3, activation='relu', border_mode='same', init='glorot_normal',
                                W_regularizer=regularizers.l2(0.0005))(unpool4)  # 10

        unpool3 = UpSampling1D(2)(deconv3)  # 11
        deconv2 = Convolution1D(32, 3, activation='relu', border_mode='same', init='glorot_normal',
                                W_regularizer=regularizers.l2(0.0005))(unpool3)  # 12

        unpool2 = UpSampling1D(2)(deconv2)  # 13
        deconv1 = Convolution1D(16, 5, activation='relu', border_mode='same', init='glorot_normal',
                                W_regularizer=regularizers.l2(0.0005))(unpool2)  # 14

        unpool1 = UpSampling1D(2)(deconv1)  # 15
        decoded = Convolution1D(3, 7, activation='sigmoid', border_mode='same', init='glorot_normal',
                                W_regularizer=regularizers.l2(0.0005))(unpool1)  # 16

        self.encoder = Model(input=input_img, output=encoded)
        self.autoencoder = Model(input=input_img, output=decoded)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        print(self.autoencoder.summary())

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/full_model_encoder.h5')
        self.autoencoder.save('./save_data/full_model_autoencoder.h5')

    def load_weights(self, ae01, ae02, ae03, ae04):
        self.autoencoder.layers[1].set_weights(ae01.layers[1].get_weights())
        self.autoencoder.layers[3].set_weights(ae02.layers[1].get_weights())
        self.autoencoder.layers[5].set_weights(ae03.layers[1].get_weights())
        self.autoencoder.layers[7].set_weights(ae04.layers[1].get_weights())

        self.autoencoder.layers[10].set_weights(ae04.layers[4].get_weights())
        self.autoencoder.layers[12].set_weights(ae03.layers[4].get_weights())
        self.autoencoder.layers[14].set_weights(ae02.layers[4].get_weights())
        self.autoencoder.layers[16].set_weights(ae01.layers[4].get_weights())


class AutoEncoderStack01(object):
    def __init__(self, data):
        input_img = Input(shape=(data.shape[1], data.shape[2]))  # 0
        conv1 = Convolution1D(16, 7, activation='relu',
                              border_mode='same')(input_img)  # 1
        pool1 = MaxPooling1D(2, border_mode='same')(conv1)  # 2

        encoded = pool1

        unpool1 = UpSampling1D(2)(pool1)  # 3
        decoded = Convolution1D(
            1, 7, activation='sigmoid', border_mode='same')(unpool1)  # 4
        crop = Cropping1D(cropping=(0, 1))(decoded)

        self.encoder = Model(input=input_img, output=encoded)
        self.autoencoder = Model(input=input_img, output=crop)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        print(self.autoencoder.summary())

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack01_encoder.h5')
        self.autoencoder.save('./save_data/stack01_autoencoder.h5')


class AutoEncoderStack02(object):
    def __init__(self, data):
        input_img = Input(shape=(data.shape[1], data.shape[2]))  # 0
        conv2 = Convolution1D(32, 5, activation='relu',
                              border_mode='same')(input_img)  # 1
        pool2 = MaxPooling1D(2, border_mode='same')(conv2)  # 2

        encoded = pool2

        unpool2 = UpSampling1D(2)(pool2)  # 3
        decoded = Convolution1D(
            16, 5, activation='linear', border_mode='same')(unpool2)  # 4

        self.encoder = Model(input=input_img, output=encoded)
        self.autoencoder = Model(input=input_img, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        print(self.autoencoder.summary())

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack02_encoder.h5')
        self.autoencoder.save('./save_data/stack02_autoencoder.h5')


class AutoEncoderStack03(object):
    def __init__(self, data):
        input_img = Input(shape=(data.shape[1], data.shape[2]))  # 0
        conv3 = Convolution1D(64, 3, activation='relu',
                              border_mode='same')(input_img)  # 1
        pool3 = MaxPooling1D(2, border_mode='same')(conv3)  # 2

        encoded = pool3

        unpool3 = UpSampling1D(2)(pool3)  # 4
        decoded = Convolution1D(
            32, 3, activation='linear', border_mode='same')(unpool3)  # 13

        self.encoder = Model(input=input_img, output=encoded)
        self.autoencoder = Model(input=input_img, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        print(self.autoencoder.summary())

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack03_encoder.h5')
        self.autoencoder.save('./save_data/stack03_autoencoder.h5')


class AutoEncoderStack04(object):
    def __init__(self, data):
        input_img = Input(shape=(data.shape[1], data.shape[2]))  # 0
        conv4 = Convolution1D(128, 3, activation='relu',
                              border_mode='same')(input_img)  # 1
        pool4 = MaxPooling1D(2, border_mode='same')(conv4)  # 2

        encoded = pool4

        unpool4 = UpSampling1D(2)(pool4)  # 3
        decoded = Convolution1D(
            64, 3, activation='linear', border_mode='same')(unpool4)  # 4

        self.encoder = Model(input=input_img, output=encoded)
        self.autoencoder = Model(input=input_img, output=decoded)
        print(self.autoencoder.summary())

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             nb_epoch=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack04_encoder.h5')
        self.autoencoder.save('./save_data/stack04_autoencoder.h5')
