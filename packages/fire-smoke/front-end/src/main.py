import sys

import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, \
    QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
from qfluentwidgets import FluentIcon, FluentWindow
from qframelesswindow import FramelessWindow, StandardTitleBar
from worker import Worker

from steel_plate_interface import SteelPlateInterface
from fire_smoke_interface import FireSmokeInterface
from person_interface import PersonInterface
from person_worker import PersonWorker
from fruits_interface import FruitsInterface
from car_interface import CarInterface
from helmet_interface import HelmetInterface
from tumor_interface import TumorInterface


class MainWindow(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.worker = Worker()
        self.worker1 = PersonWorker()

        self.fireSmokeInterface = FireSmokeInterface('火焰烟雾检测', parent=self, worker=self.worker)
        self.personInterface = PersonInterface('行人检测', parent=self, worker=self.worker1)
        self.fruitsInterface = FruitsInterface('水果检测', parent=self, worker=self.worker)
        self.carInterface = CarInterface('汽车检测', parent=self, worker=self.worker)
        self.helmetInterface = HelmetInterface('安全帽检测', parent=self, worker=self.worker)
        self.tumorInterface = TumorInterface('脑部肿瘤检测', parent=self, worker=self.worker)
        self.steelPlateInterface = SteelPlateInterface('钢材表面缺陷检测', parent=self, worker=self.worker)

        self.current_results = None

        self.init_navigation()
        self.init_window()
        self.init_listener()

    def init_navigation(self):
        self.addSubInterface(self.fireSmokeInterface, FluentIcon.FLAG, '火焰、烟雾陷检测')
        self.addSubInterface(self.personInterface, FluentIcon.ROBOT, '行人检测')
        self.addSubInterface(self.fruitsInterface, FluentIcon.LIBRARY, '水果检测')
        self.addSubInterface(self.carInterface, FluentIcon.CAR, '汽车检测')
        self.addSubInterface(self.helmetInterface, FluentIcon.HEADPHONE, '安全帽检测')
        self.addSubInterface(self.tumorInterface, FluentIcon.HOME, '脑部肿瘤检测')
        self.addSubInterface(self.steelPlateInterface, FluentIcon.HOME, '钢材表面缺陷检测')

    def init_window(self):
        self.navigationInterface.setExpandWidth(250)
        self.setTitleBar(StandardTitleBar(self))
        self.titleBar.raise_()
        self.setWindowIcon(QIcon('logo.png'))
        self.setWindowTitle("目标检测")
        self.stackedWidget.setStyleSheet("background-color: rgb(255, 255, 255);border: none;")

        # 居中布局
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.resize(w, h)
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

        # 这行代码必须在 setExpandWidth() 后面调用
        self.navigationInterface.setCollapsible(False)

    def handler_switch_to(self, current_widget):
        self.worker.jump_out = True
        self.worker.send_img.emit(np.array([]))  # 检测结果图像
        self.worker.send_statistic.emit({})
        self.worker.classes = None

        if isinstance(current_widget, FruitsInterface):
            self.fruitsInterface.init_model('fruits.pt')
            self.worker.source = "datasets/watermelon.mp4"
            self.worker.frame_show_1(self.worker.source)
        elif isinstance(current_widget, CarInterface):
            self.carInterface.init_model('car.pt')
            self.worker.source = "datasets/car.mp4"
            self.worker.frame_show_1(self.worker.source)
        elif isinstance(current_widget, HelmetInterface):
            self.helmetInterface.init_model('helmet.pt')
            self.worker.source = "datasets/helmet.mp4"
            self.worker.frame_show_1(self.worker.source)
        elif isinstance(current_widget, SteelPlateInterface):
            self.steelPlateInterface.init_model('steelplate.pt')
            self.worker.source = "datasets/person.mp4"
            self.worker.frame_show_1(self.worker.source)
        elif isinstance(current_widget, FireSmokeInterface):
            self.fireSmokeInterface.init_model('fire_smoke.pt')
            self.worker.source = "datasets/fire_smoke.avi"
            self.worker.frame_show_1(self.worker.source)
        elif isinstance(current_widget, PersonInterface):
            self.personInterface.init_model('person.pt')
            self.worker1.source = "datasets/person.mp4"
            self.worker1.frame_show_1(self.worker1.source)
        elif isinstance(current_widget, TumorInterface):
            self.tumorInterface.init_model('tumor.pt')
            self.worker.source = "datasets/fire_smoke.avi"
            self.worker.frame_show_1(self.worker.source)

        # print(self.worker)

    def init_listener(self):
        self.stackedWidget.currentChanged.connect(lambda: self.handler_switch_to(self.stackedWidget.currentWidget()))


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
