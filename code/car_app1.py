import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
from keras.models import load_model
import numpy as np
from enlighten_inference import EnlightenOnnxModel

form_window = uic.loadUiType('../car_accident_qt_s.ui')[0]

import torch
import cv2
import matplotlib.pyplot as plt
from src.Models import Unet
# 모델 로드 1
weight_path = '../models/[PART]Unet.pt'
n_classes = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
model.eval()
# 모델 로드 2
labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []
n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for label in labels:
    model_path = f'../models/[DAMAGE][{label}]Unet.pt'
    model2 = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model2.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model2.eval()
    models.append(model2)


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = model
        self.path = ('../image/111.jpg', '')
        self.btn_open.clicked.connect(self.image_open_slot)



    def image_open_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open File',
                    '../image', 'Image Files(*.jpg;*.png);;All Files(*.*)')        # 'Open File 제목', '경로', '파일 형식;전체파일형식'
        if self.path[0]:                # 무언가 입력을 받았을때
            pixmap = QPixmap(self.path[0])
            self.lbl_image.setPixmap(pixmap)
            self.lbl_image.resize(64, 64)
            try:
                img = cv2.imread(self.path[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                model_light = EnlightenOnnxModel()
                img = model_light.predict(img)
                img_input = img / 255.
                img_input = img_input.transpose([2, 0, 1])
                img_input = torch.tensor(img_input).float().to(device)
                img_input = img_input.unsqueeze(0)
                output = model(img_input)
                img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
                img_output = img_output.transpose([1, 2, 0])
                area_sum = img_output.sum()

                outputs = []
                for i, model2 in enumerate(models):
                    output = model2(img_input)
                    img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
                    img_output = img_output.transpose([1, 2, 0])
                    outputs.append(img_output)
                area_breakage = outputs[0].sum()
                area_crushed = outputs[1].sum()
                area_scratch = outputs[2].sum()
                area_seperated = outputs[3].sum()


                # 수리비
                price_table = [
                    200,  # Breakage_3 / 파손 200
                    150,  # Crushed_2 / 찌그러짐 150
                    100,  # Scratch_0 / 스크래치 100
                    200,  # Seperated_1 / 이격 200
                ]
                total = 0
                for i, price in enumerate(price_table):
                    area = outputs[i].sum()
                    total += area * price
                self.lbl_repair.setText(f'고객님, 총 수리비는 {total:,}원 입니다!')

                # 손상심각도
                severity = ( area_breakage * 3.0 + area_crushed * 2.0 + area_seperated * 1.2 + area_scratch * 1.0) * 100 / (3 * area_sum)
                if 0 <= severity < 11:
                    level = 4
                elif severity < 41:
                    level = 3
                elif severity < 81:
                    level = 2
                else:
                    level = 1
                self.lbl_level.setText('손상심각도 : {}등급'.format(level))
            except:
                print('error')




if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())