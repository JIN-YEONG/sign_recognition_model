
from util import datagenerator_split as dg   # 900개 데이터용 
from util import datagenerator as dc   # 100개 데이터 상하좌우


from keras.models import load_model, Model
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Activation, Flatten, LeakyReLU
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad, Adamax, Nadam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, l1

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from PIL import Image, ImageDraw

import numpy as np
import os
import sys


# 이미지의 좌표를 예측하는 모델


class R_model:
    def __init__(self):
        self.root_path = os.getcwd() + '/'
        
        # 900 data
        self.x_train_path = 'npy/x_train.npy'
        self.x_test_path = 'npy/x_test.npy'
        self.y_train_path = 'npy/y1_train.npy'
        self.y_test_path = 'npy/y1_test.npy'

        self.x_train = np.load(self.root_path + self.x_train_path)
        self.x_test = np.load(self.root_path + self.x_test_path)
        self.y_train = np.load(self.root_path + self.y_train_path)
        self.y_test = np.load(self.root_path + self.y_test_path)


        self.max_box = 1
        
        self.model = self.model_body()
        # self.train_model = self.cross_val()


    def model_body(self):
        # model_path = self.root_path + 'custom_max_leaky_model.h5'
        model_path = self.root_path + 'model/separ_max4_leaky_model.h5'

        base_model = load_model(model_path)

        # x = GlobalAveragePooling2D()(base_model.output)
        x = Flatten()(base_model.output)

        # output
        output = Dense(4, name ='xy',)(x)
        output = Activation('relu')(output)
        output = Reshape((self.max_box, 4))(output)

        model = Model(base_model.input, output)

        opt = SGD() # 45.7300
        # opt = Adadelta() #  67.6711
        # opt= Adam() # 10.2996
        # opt= RMSprop() # 49.5105
        # opt = Adagrad()   # 124.8453
        # opt = Adamax(learning_rate=0.05) # 11.4680
        # opt = Nadam() # 47.2392
        # opt = RMSprop() # 5.1012

        model.compile(loss='mean_absolute_error', optimizer=opt)

        # print(sys.getsizeof(model))   # 56
        return model

    def cross_val(self):
        model = self.model

        kf = KFold(n_splits=10, shuffle=True, random_state=17)
        
        earlystopping = EarlyStopping(monitor='val_loss',patience=20, mode='auto')
        model_path = 'model/RS_' + 'mae{val_loss:.4f}.h5'
        modelcheck = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, mode='auto')

        all_loss =[]
        batch = 2
        for train_index, val_index in kf.split(self.x_train):
            train_data = self.x_train[train_index]
            val_data = self.x_train[val_index]
            train_label = self.y_train[train_index]
            val_label = self.y_train[val_index]

            train_size= len(train_data)
            val_size = len(val_data)
            
            # train_generator = dg.DataGenerator(train_data, train_label, batch_size=batch)
            # val_generator = dg.DataGenerator(val_data, val_label, batch_size=batch)

            train_generator = dc.DataGenerator(train_data, train_label, batch_size=batch)
            val_generator = dc.DataGenerator(val_data, val_label, batch_size=batch)


            model.fit_generator(
                train_generator,
                steps_per_epoch=max(1, train_size // batch),
                epochs=100,
                validation_data = val_generator,
                validation_steps= max(1, val_size//batch),
                callbacks=[earlystopping, modelcheck]
            )

            test_size = len(self.x_test)
            
            # test_generator = dg.DataGenerator(self.x_test,self.y_test, batch_size=batch)

            test_generator = dc.DataGenerator(self.x_test,self.y_test, batch_size=batch)
            
            val_loss = model.evaluate_generator(test_generator, steps=max(1, test_size//batch))

            all_loss.append(val_loss)

        print(np.mean(all_loss))

        return model

    
    def check(self):
        print(self.img_path_data[:5])
        print(self.xywh_data[:5])


if __name__ == '__main__':
    r = R_model()
    r.cross_val()
    # r.check()
    