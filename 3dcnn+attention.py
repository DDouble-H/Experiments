import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import GRU, Bidirectional, LSTM
from keras.layers import BatchNormalization
from keras.layers import Reshape, Dropout, Dense, Flatten, Conv3D, Activation, AveragePooling3D, Permute

from keras import models
from keras import layers


def tdcnn(main_input, time):
    pool_size1 = (1, 4, 2)
    pool_size2 = (1, 2, 4)

    activation_name = 'elu'

    # Input
    main_batch_norm = BatchNormalization()(main_input)

    conv3d = Conv3D(kernel_size=(5, 4, 2), strides=(1, 1, 1), filters=128, padding='same')(main_input)
    batch_norm = BatchNormalization()(conv3d)
    conv = Activation(activation_name)(batch_norm)

    up_sample1 = AveragePooling3D(pool_size=pool_size1, strides=(1, 1, 1), padding='same')(conv)
    up_sample2 = AveragePooling3D(pool_size=pool_size2, strides=(1, 1, 1), padding='same')(conv)
    conv_concat = concatenate([main_batch_norm, conv, up_sample1, up_sample2])

    conv_1d = Conv3D(kernel_size=(1, 1, 1), strides=(1, 1, 1), filters=16)(conv_concat)
    batch_norm = BatchNormalization()(conv_1d)
    activation = Activation(activation_name)(batch_norm)

    bidir_rnn = Reshape((time, -1))(activation)
    bidir_rnn1 = Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(bidir_rnn)
    bidir_rnn2 = Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(bidir_rnn1)

    # bidir_rnn1 = Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(bidir_rnn)
    # bidir_rnn2 = Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(bidir_rnn1)

    permute = Permute((2, 1))(bidir_rnn2)
    dense_1 = Dense(X_train.shape[1:], activation='softmax')(permute)
    prob = layers.Permute((2, 1), name='attention_vec')(dense_1)
    attention_mul = layers.multiply([main_input, prob])
    attention_mul = layers.Flatten()(attention_mul)
    flat = Flatten()(attention_mul)
    drop = Dropout(0.5)(flat)
    dense_2 = Dense(200)(drop)
    main_output = Dense(2, activation='softmax')(dense_2)

    return main_output

X_train, X_test, Y_train, Y_test = '', '', '', ''

main_input = Input(shape=(X_train.shape[1:]), dtype='float32', name='main_input')
model = tdcnn(main_input, X_train.shape[1])
model = Model(inputs=main_input, output=model)
optimizer = keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',
                                                                             tf.keras.metrics.Precision(name='precision'),
                                                                             tf.keras.metrics.Recall(name='recall')])

model.summary()