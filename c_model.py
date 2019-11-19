# 정말 여러개의 라벨이 필요
# 현재 1개


from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Reshape,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split, KFold

from util import datagenerator_split as ds

import numpy as np
import os
import sys

class C_model:
    def __init__(self):
        self.root_path = os.getcwd() + '/'

        self.x_train_path = 'npy/x_train.npy'
        self.x_test_path = 'npy/x_test.npy'
        self.y_train_path = 'npy/y2_train.npy'
        self.y_test_path = 'npy/y2_test.npy'

        self.label_path = 'npy/label_encode_list.npy'

        self.x_train = np.load(self.root_path + self.x_train_path)
        self.x_test = np.load(self.root_path + self.x_test_path)
        self.y_train = np.load(self.root_path + self.y_train_path)
        self.y_test = np.load(self.root_path + self.y_test_path)

        self.label_list = np.load(self.root_path + self.label_path)

        self.max_box = 1
        self.num_class = len(self.label_list)

        self.model = self.model_body()

        # self.print_value()

    def print_value(self):
        print('x_train.shape', self.x_train.shape)
        print('x_test.shape', self.x_test.shape)
        print('y_train.shape', self.y_train.shape)
        print('y_test.shape', self.y_test.shape)

        print('label_list.shape', self.label_list.shape)

        print('num_class', self.num_class)


    def model_body(self):
        # model_path = self.root_path + 'l2.00_max4_leaky_model.h5'
        model_path = self.root_path + 'model/separ_max4_leaky_model.h5'
        base_model = load_model(model_path)

        x = GlobalAveragePooling2D()(base_model.output)
        # x = Flatten()(base_model.output)

        output = Dense(self.num_class, name='xy')(x)
        output = Activation('sigmoid')(output)
        # output = Reshape((self.max_box, 2))(output)

        model = Model(base_model.input, output)

        model.compile(loss = 'binary_crossentropy', optimizer='adam')

        return model
    

    def croos_val(self):
        model = self.model

        kf = KFold(n_splits=5, shuffle=True, random_state=17)

        earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        model_path = 'model/1028/CS_' + 'mae{val_loss: .4f}.h5'
        modelcheck = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='auto')

        all_loss=[]
        batch = 2
        for train_index, val_index in kf.split(self.x_train):
            train_data = self.x_train[train_index]
            val_data = self.x_train[val_index]
            train_label = self.y_train[train_index]
            val_label = self.y_train[val_index]

            train_size = len(train_data)
            val_size = len(val_data)

            train_generator = ds.DataGenerator(train_data, train_label, batch_size= batch)
            val_genterator = ds.DataGenerator(val_data, val_label, batch_size=batch)

            model.fit_generator(
                train_generator,
                steps_per_epoch=max(1,train_size // batch),
                epochs = 50,
                validation_data = val_genterator,
                validation_steps= max(1,val_size/ batch),
                callbacks=[earlystopping, modelcheck]
            )


            test_size = len(self.x_test)

            test_generator = ds.DataGenerator(self.x_test, self.y_test, batch_size=batch)

            val_loss = model.evaluate_generator(test_generator, steps=max(1, test_size//batch))

            all_loss.append(val_loss)

        print(np.mean(all_loss))

        

if __name__ == '__main__':
    c = C_model()
    c.croos_val()

    