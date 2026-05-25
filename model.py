#Данный файл содержит непосредственно описание модели

#CNN для распознавания дипфейков
#Архитектура для класса RotEyes:
#  Фото  -> CNN(1) -> Выделение области глаз, глазного яблока и радужки в три потока -> CNN по области глаз(2)(4 слоя) -> активация, субдескритизация, выпрямление, полносвязный -> 
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

import torch
import torch.nn as nn
import math
import torchvision
import dlib

IMGWIDTH=256
IMGHEIGHT=128

#Выделение массива с тремя областями из входного изображения
def see_eyes():
	

##############################################
#Область глаза, три ветви, 4 слоя
class RotEyes(nn.Module):
	def __init__(self, num_classes=1):
		super().__init__()
		self.RotArea=self._branch()
		self.RotEye=self._branch()
		self.RotIris=self._branch()

	self.fusion=nn.Sequental(
			nn.Linear(256 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
		)
	
	def _branch(self):
		reurn nn.Sequental(
			#256x128 -> 128x64
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #128x64 -> 64x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #64x32 -> 32x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            #32x16 -> 16x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),

			nn.AdaptiveAvgPool2d((1, 1))  
	)
	
	def forward(self, area: torch.Tensor, eye_l: torch.Tensor, eye_r: torch.Tensor, iris_l: torch.Tensor, iris_r: torch.Tensor):
		f_area = self.RotArea(area).flatten(start_dim=1)
		
        f_eye_l  = self.RotEye(eye_l).flatten(start_dim=1)
		f_eye_r  = self.RotEye(eye_r).flatten(start_dim=1)
        f_iris_l  = self.RotIris(iris_l).flatten(start_dim=1)
        f_iris_r  = self.RotIris(iris_r).flatten(start_dim=1)
        
		f_eye  = (f_eye_l  + f_eye_r)  * 0.5
   		f_iris = (f_iris_l + f_iris_r) * 0.5
        final = torch.cat([f_area, f_eye, f_iris], dim=1)  
        return self.fusion(final)                        

##############################################
#Макет, на всю область глаз, 6 слоёв
class RotCNN6(nn.Module):
	def __init__(self, num_classes=1):
	super().__init__()
	self.features nn.Sequential(
            #256x128 -> 128x64
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            #128x64 -> 64x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            #64x32 -> 32x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            #32x16 -> 32x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
			#32x16 -> 16x8
			nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),	nn.MaxPool2d(2),
			#16x8 -> 16x8
			nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
					
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  
        x = self.classifier(x)
        return x
##############################################

#Макет, на всю область глаз, 4 слоя
class RotCNN4(nn.Module):
	def __init__(self, num_classes=1):
	super().__init__()
	self.features nn.Sequential(
            #256x128 -> 128x64
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            #128x64 -> 64x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
			#64x32 -> 32x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            #32x16 -> 16x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
											
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
##############################################
