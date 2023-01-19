import sys # 파이썬 기본 라이브러리
from PyQt5.QtWidgets import * # *주면 모든파일 import함
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from keras.models import load_model
import numpy as np
import cv2 #pip install opencv-python 써서 설치
import time

import torch
import cv2
import matplotlib.pyplot as plt

from src.Models import Unet
weight_path = '../models/[DAMAGE][Scratch_0]Unet.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = 2

form_window = uic.loadUiType('./mjj_cat_and_dog.ui')[0] # loadUiType이걸 쓰면 class로 만들어줌

class Exam(QWidget, form_window): # 다중 상속 가능 # QWidget 닫기 최소화 등 버튼 활성화
    def __init__(self):
        super().__init__()
        self.setupUi(self) # setupUi가 많은 것을 설정해줌
        self.model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
        self.model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
        self.path = ('../sample/damage/0000006_as-0036229.jpg', '')
        self.btn_open.clicked.connect(self.image_open_slot)


    def image_open_slot(self):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        flag = True
        while flag:
            _, frame = capture.read()
            cv2.imshow('VideoFrame', frame)
            cv2.imwrite('./capture.png', frame)
            time.sleep(2)
            key = cv2.waitKey(33) # 키 입력을 기다린다.
            if key == 27:
                flag = False
            pixmap = QPixmap('./capture.png')
            self.lbl_image.setPixmap(pixmap)
            try:
                img = Image.open('./capture.png')
                img = img.convert('RGB')
                img = img.resize((64, 64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1, 64, 64, 3)
                pred = self.model.predict(data)
                print(pred)
                # if pred < 0.5:
                #     self.lbl_pred.setText('고양이일 확률이 {}% 입니다.'.format(
                #         ((1 - pred[0][0]) * 100).round(1)
                #     ))
                # else:
                #     self.lbl_pred.setText('강아지일 확률이 {}% 입니다.'.format(
                #         ((pred[0][0]) * 100).round(1)
                #     ))

            except:
                print('error')

if __name__ == "__main__": # 나중에 모듈로 써먹기 위해서
    app = QApplication(sys.argv) # 어플이 어플을 동작하게 하는 기능
    mainWindow = Exam() # 객체는 여기서 만들어 진다.
    mainWindow.show() # 화면에 출력해라
    sys.exit(app.exec_()) # 사용자가 한 액션을 처리하는 것 # exec 무한루프 # 윈도우 종료시 exit
    # 클릭 한 후 시그널이 발생하며 받을 슬롯을 지정해준다.