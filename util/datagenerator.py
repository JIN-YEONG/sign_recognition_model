import numpy as np
import keras

from keras.models import Model
from keras.layers import Dense,Input, Reshape, Flatten

from PIL import Image
from PIL import ImageDraw

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, label_1, batch_size=1, dim=(608, 608), n_channels=3, n_classes=2, max_box=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.label_1 = label_1
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_box = max_box
        self.on_epoch_end()

    def __len__ (self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_label_1_temp = [self.label_1[k] for k in indexes]


        x, y1 = self.__data_generation(list_IDs_temp, list_label_1_temp)

        return x, y1


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def im_show(self, img, label):
        # img_t = Image.fromarray(img)

        # label : (x,y, w,h)
        draw = ImageDraw.Draw(img)
        label = np.array(label)
        # print(label.shape)

        x = label[0]
        y = label[1]
        w = label[2]
        h = label[3]
        
        hf_w = w // 2 
        hf_h = h // 2

        xmin = x - hf_w
        ymin = y - hf_h

        xmax = x + hf_w
        ymax = y + hf_h

        
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)

        img.show()


    def __data_generation(self, list_IDs_temp, list_label_1_temp):
        # x = np.empty((self.batch_size*4, *self.dim, self.n_channels))
        # y1 = np.empty((self.batch_size*4, self.max_box, 4))
        # y2 = np.empty((self.batch_size*4, self.max_box, self.n_classes))
        x =[]
        y1=[]


        for (img, label1) in zip(list_IDs_temp, list_label_1_temp):
            # list_label_1_temp = (x,y,w,h)
            
            # 원본
            origin_img = Image.open(img)   # (N, 608,608,3)
            lr_img = origin_img.transpose(Image.FLIP_LEFT_RIGHT)
            tb_img = origin_img.transpose(Image.FLIP_TOP_BOTTOM)
            lr_tb_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM) 

            origin_img = np.array(origin_img) / 255.
            lr_img = np.array(lr_img) / 255.
            tb_img = np.array(tb_img) / 255.
            lr_tb_img = np.array(lr_tb_img) / 255.

            origin_label_1_list = []


            lr_label_1_list = []


            tb_label_1_list = []


            lr_tb_label_1_list = []


            for i in range(self.max_box):
                # label 지정
                # 원본
                origin_label_1 = label1[i]   # (4,)


                lr_label_1 = np.zeros(4)
                lr_label_1[:] = origin_label_1    


                tb_label_1 = np.zeros(4)
                tb_label_1[:] = origin_label_1


                lr_tb_label_1 = np.zeros(4)
                lr_tb_label_1[:] = origin_label_1

                # print(origin_label_1)
                # print(lr_label_1)
                # print(tb_label_1)
                # print(lr_tb_label_1)

                lr_label_1[0] = self.dim[0] - origin_label_1[0]

                tb_label_1[1] = self.dim[0] - origin_label_1[1]

                lr_tb_label_1[0] = self.dim[0] - origin_label_1[0]
                lr_tb_label_1[1] = self.dim[0] - origin_label_1[1]
                # print(origin_label_1)
                # print(lr_label_1)
                # print(tb_label_1)
                # print(lr_tb_label_1)


                origin_label_1_list.append(origin_label_1)

                lr_label_1_list.append(lr_label_1)


                tb_label_1_list.append(tb_label_1)


                lr_tb_label_1_list.append(lr_tb_label_1)



                #######################################################################
                # self.im_show(origin_img, origin_label_1)
                # self.im_show(lr_img, lr_label_1)
                # self.im_show(tb_img, tb_label_1)
                # self.im_show(lr_tb_img, lr_tb_label_1)


            x.append(origin_img)
            x.append(lr_img)
            x.append(tb_img)
            x.append(lr_tb_img)

            y1.append(origin_label_1_list)
            y1.append(lr_label_1_list)
            y1.append(tb_label_1_list)
            y1.append(lr_tb_label_1_list)


        x = np.array(x)
        y1 = np.array(y1) #/ self.dim[0]


        # # print('------------------------------------------------')
        # print(x)
        # print(x.shape)   # (4, 608, 608, 3)
        # # print('------------------------------------------------')
        # print(y1)
        # print(y1.shape)   # (4, 1, 4)
        # # print('------------------------------------------------')
        # # print(y2)
        # print(y2.shape)   # (4, 1, 2)
        

        return x, y1


    



'''
if __name__ == '__main__':
    ann_path = 'train.txt'
    max_boxs = 1
    class_num = 1
    input_shape = 608
    # data 

    box_data=[]
    img_data =[]

    with open(ann_path) as an:
        lines = an.readlines()

    for line in lines:
        split_line = line.split()


        img_data.append(split_line[0])


        box = np.zeros((max_boxs, 5))  
        for i, boxs in  enumerate(split_line[1:]):
            split_box = list(map(int,boxs.split(',')))
            box[i] = split_box  
            # print(box)

            box_data.append(box)   

    img_data = np.array(img_data)
    box_data = np.array(box_data)
    # print(box_data)
    # print(img_data.shape, box_data.shape)   # (100, 608, 608, 3) (100, 1, 5)



    # new data
    y_box_data=[]
    y_cl_data=[]

    for i, data in enumerate(box_data):
        t_box_data = np.zeros((max_boxs, 4))
        t_cl_data = np.zeros((max_boxs, 1+class_num), dtype= 'int')
        # print(t_box_data.shape) # 7
        for i, d in enumerate(data):
            z_box_data = np.zeros(4)  # class_num=1
            z_cl_data = np.zeros((1+class_num))  # class_num=1

            # print(d[0:4])
            xy = (d[2:4] + d[0:2]) // 2
            wh = d[2:4] - d[0:2]

            z_box_data[0:2] = xy #/ input_shape
            z_box_data[2:4] = wh #/ input_shape
            
            # print(z_box_data)
            
            c = int(d[-1])
            z_cl_data[0+c] = 1

            # print(z_box_data)
            t_box_data[i] = z_box_data
            t_cl_data[i] = z_cl_data

        y_box_data.append(t_box_data)
        y_cl_data.append(t_cl_data)

    y_box_data = np.array(y_box_data)
    y_cl_data = np.array(y_cl_data)
    
    # print(y_box_data)

    gd = DataGenerator(img_data, y_box_data, y_cl_data)

    input_l = Input(shape=(608,608,3))
    
    base = Flatten()(input_l)

    output1 = Dense(4)(base)
    output1 = Reshape((max_boxs,4))(output1)

    output2 = Dense(2)(base)
    output2 = Reshape((max_boxs, 1+class_num))(output2)

    model = Model(input_l, [output1, output2])

    model.compile(loss ='mse',optimizer='adam')
    model.fit_generator(gd, steps_per_epoch=100 ,epochs=100)

'''


