#Данный файл содержит непосредственно описание модели

#CNN для распознавания дипфейков
#Архитектура для классов RotIris, RotBall и RotEye:
#  Фото  -> CNN(1) -> Выделение области глаз и радужки в два потока -> CNN по области глаз(2)(4 слоя) -> активация, субдескритизация, выпрямление, полносвязный -> 
#                                                                   -> CNN по радужке(2)(4 слоя)      -> активация, субдескритизация, выпрямление, полносвязный -> 
#        -> Если на входе видео, то проходим в несколько итераций, по полученному массиву значений, ReLU с bias=0,7 и (3)-> выводим на градиентный спуск два полученых значения и полученный (3), получаем результат
# (1) CNN обучается распознавать по области глаз и радужки отдельно
# (2) Обучение на итоговой выборке, нужна вероятность того, что данное изображение принадлежит дипфейку
# (3) Для видео мы можем получить динамику EAR, потому мы можем, по сути создать эрзац-свёрточную функцию, в которой мы применяем метод скользящего окна на 20 значений через Max Pooling, с шагом в 5, и подаём на Байесовский
#    обучение с учителем без заранее размеченных параметров

# Архитектура для класса RotCNN
#Фото  -> CNN(1) -> Выделение области глаз -> CNN по области глаз(2)(6 слоёв) -> активация, субдескритизация, выпрямление, полносвязный ->                           
#        -> Если на входе видео, то проходим в несколько итераций, отбрасывая кадры с EAR<0.25 и (3)-> выводим на градиентный спуск два полученых значения: вероятность для глаз(3), получаем результат

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

IMGWIDTH=256
IMGHEIGHT=128

#Выделение массива с тремя областями из входного изображения
def see_eyes()

#Область глаза
class RotEye(nn.Module):
	def __init__(self, learning_rate=0.001, num_classes=2):
	
#Зона, ограниченная веками
class RotBall(nn.Module):
	def __init__(self, learning_rate=0.001, num_classes=2):
	
#Зона радужки
class RotIris(nn.Module):
	def __init__(self, learning_rate=0.001, num_classes=2):
	
#Макет на всю область глаз
class RotCNN(nn.Module):
	def __init__(self, learning_rate=0.001, num_classes=2,):
	self.relu=nn.ReLU()
	self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
	def init_model(self):
	x=Input(shape=(IMGWIDTH,IMGHEIGHT,3))
	conv1=nn.Conv2d(64,(6,6),padding=0, strides=1)(x) #256 - 128 -3
	conv1=nn.BatchNorm2d(8)(conv1)
	conv1=self.relu(conv1)
	
	conv2=nn.Conv2d(32,(3,3),padding=1, strides=1)(conv1)
	conv2=nn.BatchNorm2d(8)(conv2)
	conv2=self.relu(conv2)
	
	conv3=nn.Conv2d(32,(3,3),padding=2, strides=1)(conv2)
	conv3=nn.BatchNorm2d(6)(conv3)
	conv3=self.relu(conv3)
	
	conv4=nn.Conv2d(16,(2,2),padding=1, strides=1)(conv3)
	conv4=nn.BatchNorm2d(6)(conv4)
	conv4=self.relu(conv4)
	
	conv5=nn.Conv2d(8,(2,2),padding=1, strides=1)(conv4)
	conv5=nn.BatchNorm2d(4)(conv5)
	conv5=self.relu(conv5)
	
	conv6=nn.Conv2d(4,(2,2),padding=2, strides=1)(conv5)
	conv6=nn.BatchNorm2d(4)(conv6)
	conv6=self.relu(conv6)
	
	nn.MaxPool2d(kernel_size=(2, 2))(y)
	nn.Dropout2d(0.5)(y)
	
	        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(negative_slope=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)
