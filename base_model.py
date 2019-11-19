import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Activation, UpSampling2D, Reshape, MaxPool2D, SeparableConv2D
from keras.regularizers import l2

class Custom_model:
    def __init__(self):
        self.input_size = 608
        self.input_dim = 3
        self.input_layer = self.start_layer()
        self.model = self.model_body()

    def start_layer(self):
        input_shape = (self.input_size, self.input_size, self.input_dim)

        return Input(shape = input_shape)

    def base_layer(self, input_layer, output, size=3, stride=1, lr=0.01):
        # x = Conv2D(output, (size, size), padding = 'same', strides=stride, kernel_regularizer=l2(l=lr))(input_layer)
        x = SeparableConv2D(output, (size, size), padding = 'same', strides=stride, kernel_regularizer=l2(l=lr))(input_layer)
        # x = Activation('relu')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        return x

    def stack_base_layer(self, x, output, n):
        for i in range(n):
            x = self.base_layer(x, output, 1, 1)
            x = self.base_layer(x, output * 2, 3, 1)

        return x

    def model_body(self):
        input_l = self.input_layer

        x = self.base_layer(input_l, 16)
        x = MaxPool2D(pool_size=(2,2))(x) #304

        x = self.base_layer(x, 32)
        x = self.base_layer(x, 32)
        # x = self.base_layer(x ,32)
        x = MaxPool2D(pool_size=(2,2))(x) #152

        x = self.base_layer(x, 64)
        x = self.base_layer(x, 64)
        # x = self.base_layer(x, 64)
        x = MaxPool2D(pool_size=(2,2))(x) # 76

        x = self.base_layer(x, 128)
        x = self.base_layer(x, 128)
        # x = self.base_layer(x, 128)
        x= MaxPool2D(pool_size=(2,2))(x) # 38

        x = self.base_layer(x, 256)
        x = self.base_layer(x, 256)
        # x = self.base_layer(x, 256)
        x = MaxPool2D(pool_size=(2,2))(x) #19

        x = self.base_layer(x,512)
        x = self.base_layer(x,512)
        # x = self.base_layer(x,512)
        # x = self.base_layer(x,128)

        
        model = Model(input=input_l, output=x)

        model.summary()

        return model

    def model_save(self):
        model = self.model
        model.save('separ_max4_leaky_model.h5')
        print('save separ_max4_leaky_model.h5')

if __name__ == '__main__':
    y = Custom_model()
    y.model_save()
    



