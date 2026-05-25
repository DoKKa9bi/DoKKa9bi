#Данный файл содержит непосредственно описание модели

#CNN для распознавания дипфейков
#Архитектура для классов RotIris, RotBall и RotEye:
#  Фото  -> CNN(1) -> Выделение области глаз и радужки в два потока -> CNN по области глаз(2)(4 слоя) -> активация, субдескритизация, выпрямление, полносвязный -> 
#                                                                   -> CNN по радужке(2)(4 слоя)      -> активация, субдескритизация, выпрямление, полносвязный -> 
#																	-> CNN по области глаза(2)(4 слоя)-> активация, субдескритизация, выпрямление, полносвязный -> 
#        -> Если на входе видео, то проходим в несколько итераций, по полученному массиву значений, ReLU с bias=0,7 и (3)-> выводим на градиентный спуск три полученых значения и полученный (3), получаем результат
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

#Область глаза, три ветви, 6 слоёв
class RotEyes(nn.Module):
	def __init__(self, num_classes=1):
		super().__init__()
		self.RotArea=self.__make_branch()
		self.RotEye=self.__make_branch()
		self.RotIris=self.__make_branch()

		
#Макет, на всю область глаз, 6 слоёв
class RotCNN(nn.Module):
	def __init__(self, num_classes=1):
	super().__init__()
	self.features = nn.Sequential(
            #256x128 -> 128x64
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #128x64 -> 64x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #64x32 -> 32x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #32x16 -> 32x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)

			#32x16 -> 16x8
			nn.Conv2d(256, 256, 3, padding=1)
			nn.GroupNorm2d(256)
			nn.ReLU(inplace=True)	
			nn.MaxPool2d(2)
		
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x
