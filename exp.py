import os
import numpy as np
import h5py

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from keras.callbacks import Callback


class ClassificationReportCallback(Callback):

    def __init__(self, validation_data=(), info={}):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.infomation = info

    def on_train_begin(self, logs={}):
        self.epoch, self.val_loss = [], []
        self.val_tp, self.val_fp, self.val_tn, self.val_fn = [], [], [], []
        self.val_recall, self.val_precision, self.val_f1score, self.val_roc_auc = [], [], [], []

        self.val_pr_curve_fpr, self.val_pr_curve_tpr, self.val_pr_curve_thresholds = [], [], []
        self.val_roc_curve_fpr, self.val_roc_curve_tpr, self.val_roc_curve_thresholds = [], [], []

    def on_train_end(self, logs=None):

        if self.infomation['save']:
            self.report()
            return
        else:
            return

    def report(self):

        file_name = self.infomation['save_file_path'] + self.infomation['save_file_name'] + '.xlsx'
        import pandas as pd

        df = pd.DataFrame({
            'epoch':self.epoch,
            'true_positive':self.val_tp,
            'false_positive':self.val_fp,
            'true_negative':self.val_tn,
            'false_negative':self.val_fn,
            'recall':self.val_recall,
            'precision':self.val_precision,
            'f1_score_macro':self.val_f1score,
            'roc_auc_macro':self.val_roc_auc,
            'roc_fpr':self.val_roc_curve_fpr,
            'roc_tpr': self.val_roc_curve_tpr,
            'roc_thresholds':self.val_roc_curve_thresholds,
            'pr_fpr': self.val_pr_curve_fpr,
            'pr_tpr': self.val_pr_curve_tpr,
            'pr_thresholds': self.val_pr_curve_thresholds
        })

        df.to_excel(file_name)

    def on_epoch_end(self, epoch, logs={}):

        if np.array(self.y_val).ndim == 2:
            val_label = np.argmax(self.y_val, axis=1)
        else:
            val_label = self.y_val

        val_predict = np.argmax(self.model.predict(self.X_val), axis=1)

        tn, fp, fn, tp = confusion_matrix(val_label, val_predict).ravel()

        pre = precision_score(val_label, val_predict, average='macro')
        rec = recall_score(val_label, val_predict, average='macro')
        f1 = f1_score(val_label, val_predict, average='macro')
        auc = roc_auc_score(val_label, val_predict, average='macro')

        self.epoch.append(epoch)

        self.val_tp.append(tp)
        self.val_fp.append(fp)
        self.val_tn.append(tn)
        self.val_fn.append(fn)

        self.val_recall.append(rec)
        self.val_precision.append(pre)
        self.val_f1score.append(f1)
        self.val_roc_auc.append(auc)

        fpr, tpr, thresholds = roc_curve(val_label, val_predict)

        self.val_roc_curve_fpr.append(fpr)
        self.val_roc_curve_tpr.append(tpr)
        self.val_roc_curve_thresholds.append(thresholds)

        fpr, tpr, thresholds = precision_recall_curve(val_label, val_predict)
        self.val_pr_curve_fpr.append(fpr)
        self.val_pr_curve_tpr.append(tpr)
        self.val_pr_curve_thresholds.append(thresholds)

        return

def get_data(path, file):
    print(path)
    print(file)
    f = h5py.File(os.path.join(path, file), 'r')
    # print(f.keys())

    # healthy
    healthy_data = f['healthy_data']
    healthy_label = f['healthy_label']
    healthy_time = f['healthy_time']
    # seizure
    seizure_data = f['seizure_data']
    seizure_label = f['seizure_label']
    seizure_time = f['seizure_time']

    healthy_data = np.array(healthy_data)
    healthy_label = np.array(healthy_label)
    healthy_time = np.array(healthy_time)

    seizure_data = np.array(seizure_data)
    seizure_label = np.array(seizure_label)
    seizure_time = np.array(seizure_time)

    # shape : (data, time, ch1, ch2, freq)

    data = np.concatenate((healthy_data, seizure_data))
    label = np.concatenate((healthy_label, seizure_label))
    time = np.concatenate((healthy_time, seizure_time))

    #   print(data.shape, label.shape)
    return data, label

import keras
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import Bidirectional, LSTM
from keras.layers import BatchNormalization
from keras.layers import Reshape, Dropout, Dense, Flatten, Conv3D, Activation, AveragePooling3D, Permute, multiply

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def model1(main_input, time):
    # 3d cnn
    pool_size1 = (1, 4, 2)
    pool_size2 = (1, 2, 4)

    activation_name = 'elu'

    # Input
    main_batch_norm = BatchNormalization()(main_input)

    conv3d = Conv3D(kernel_size=(4, 5, 2), strides=(1, 1, 1), filters=128, padding='same')(main_input)
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

    flat = Flatten()(bidir_rnn2)
    drop = Dropout(0.5)(flat)
    dense = Dense(200)(drop)
    main_output = Dense(2, activation='softmax')(dense)

    return main_output

def model2(main_input, time):
    # 3d cnn + attention
    # shape : (data, time, ch1, ch2, freq)
    pool_size1 = (1, 4, 2)
    pool_size2 = (1, 2, 4)

    activation_name = 'elu'

    # Input
    main_batch_norm = BatchNormalization()(main_input)

    conv3d = Conv3D(kernel_size=(4, 5, 2), strides=(1, 1, 1), filters=128, padding='same')(main_input)
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
    dense_1 = Dense(main_input, activation='softmax')(permute)
    prob = Permute((2, 1), name='attention_vec')(dense_1)
    attention_mul = multiply([main_input, prob])
    attention_mul = Flatten()(attention_mul)
    flat = Flatten()(attention_mul)
    drop = Dropout(0.5)(flat)
    dense_2 = Dense(200)(drop)
    main_output = Dense(2, activation='softmax')(dense_2)

    return main_output

def train(patient, path, train_files, test_files):
    X_train, Y_train = get_data(path, train_files[0])

    if len(train_files) > 1:
        for file in train_files[1:]:
            Xt, Yt = get_data(path, file)
            X_train = np.concatenate((X_train, Xt))
            Y_train = np.concatenate((Y_train, Yt))

    X_test, Y_test = get_data(path, test_files)

    print(X_train.shape, X_test.shape)
    main_input = Input(shape=(X_train.shape[1:]), dtype='float32', name='main_input')
    model = model1(main_input, X_train.shape[1])
    model = Model(inputs=main_input, output=model)

    opt_adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])

    # Callback
    information = {
        'save': True,
        'save_file_path': '/Gdrive/Colab Notebooks/whhhhh/3DCNN/',
        'save_file_name': patient + test_files,
    }

    CR = ClassificationReportCallback(validation_data=(X_test, Y_test), info=information)

    histroy = model.fit(X_train, Y_train, batch_size=8, epochs=100, validation_split=0.2, verbose=1)

    del model
    K.clear_session()

    if __name__ == '__main__':
        patients = ['chb03', 'chb06', 'chb10']
        for patient in patients:
            path = '/Gdrive/Data/WHH/30/{}'.format(patient)
            patient_file = os.listdir(path)
            for i in range(7):
                train_files = patient_file
                test_files = patient_file[i]

                train_files.remove(patient_file[i])
                print(train_files, test_files)
                train(patient, path, train_files, test_files)