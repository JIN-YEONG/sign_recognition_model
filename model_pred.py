
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import cv2

import numpy as np
import os
import glob
import sys


class Pred:
    def __init__(self):
        self.root_path = os.getcwd() + '/'

        self.r_model_path_2 = 'model/RS_mae5.1012.h5'
        self.c_model_path = 'model/CS_mae 0.0006.h5'


        self.r_model_2 = self.model_load(self.r_model_path_2)
        self.c_model = self.model_load(self.c_model_path)

        self.pred_data_path = 'npy/x_test.npy'
        self.x_pred = self.load_data(self.pred_data_path)

        self.label_path = 'npy/label_encode_list.npy'
        self.label = self.label_load(self.label_path)


    def model_load(self, path):
        model_path = self.root_path + path
        # print(model_path)

        model = load_model(model_path)

        return model


    def load_data(self,path):
        
        self.file_path = np.load(path)
        # print(self.file_path)

        x_pred = []

        for data in self.file_path:
            # print(data)
            p_img = Image.open(data)#.convert('L')
            # print(sys.getsizeof(p_img))

            re_p_img = p_img.resize((608,608))
            # re_p_img.show()
            re_p_img = np.array(re_p_img)/ 255
            # re_p_img = np.reshape(re_p_img,(608,608,1))
            # print(sys.getsizeof(re_p_img))

            x_pred.append(re_p_img)
            
        x_pred = np.array(x_pred)
        # print(x_pred.shape)

        return x_pred

    def label_load(self, path):
        label_path = self.root_path + path

        label = np.load(label_path)

        label_array = []
        for data in label:
            label_array.append(data)

        label_array = np.array(label_array)

        return label_array



    def pred(self):
            r_model_2 = self.r_model_2
            c_model = self.c_model

            font_size = 20
            font = ImageFont.truetype('font/NanumGothicBold.ttf', font_size)

            y_pred_2 = r_model_2.predict(self.x_pred)
            print(y_pred_2)

            c_pred = c_model.predict(self.x_pred)
            # self.x_pred.show()
            c_pred = c_pred*100
            print(c_pred)

            index_c_pred = np.argmax(c_pred, axis=1)
            print(index_c_pred)

            label_pred = []
            for i in index_c_pred:
                label_pred.append(self.label[i])
            print(label_pred)

            # print(index_c_pred[0])
            # print(c_pred[0,index_c_pred[0]])

            for i, d in enumerate(y_pred_2):
                # print(d.shape)
                im = Image.open(self.file_path[i])
                im = im.resize((608,608))
                draw = ImageDraw.Draw(im)
   
                for data in d:

                    x = data[0]
                    y = data[1]
                    w = data[2]
                    h = data[3]
                    
                    hf_w = w // 2 
                    hf_h = h // 2

                    xmin = x - hf_w
                    ymin = y - hf_h
                    xmax = x + hf_w
                    ymax = y + hf_h


                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='blue', width=3)
                    
                    text = label_pred[i] + ': ' + '{0:.2f}%'.format(c_pred[i, index_c_pred[i]]) 
                    draw.text((xmin, ymin-(font_size + 3)), text, (255,0,0), font=font)
                    

                im.show()    



if __name__ == '__main__':
    p = Pred()
    p.pred()
    