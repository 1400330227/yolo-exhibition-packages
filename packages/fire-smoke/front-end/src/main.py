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


class MainWindow(FluentWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.worker = Worker()


        # self.layout = QVBoxLayout(self)
        # self.layout.setContentsMargins(20, 40, 20, 20)
        self.steelPlateInterface = SteelPlateInterface('钢材表面缺陷检测',parent=self, worker=self.worker)
        self.fireSmokeInterface = FireSmokeInterface('火焰烟雾陷检测',parent=self, worker=self.worker)
        # self.layout.addWidget(self.mainInterface)
        self.current_results = None

        self.init_navigation()
        self.init_window()
        self.init_listener()

    def init_navigation(self):
        self.addSubInterface(self.steelPlateInterface, FluentIcon.HOME, '钢材表面缺陷检测')
        self.addSubInterface(self.fireSmokeInterface, FluentIcon.FLAG, '火焰烟雾陷检测')

    def init_window(self):
        self.navigationInterface.setExpandWidth(250)
        self.setTitleBar(StandardTitleBar(self))
        self.titleBar.raise_()
        self.setWindowIcon(QIcon('logo.png'))
        self.setWindowTitle("目标检测")

        # 居中布局
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.resize(w, h)
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)


    def handler_switch_to(self, current_widget):
        # print(current_widget)
        # print(self.worker)
        self.worker.jump_out = True
        self.worker.source = None
        self.worker.send_img.emit(np.array([]))  # 检测结果图像
        self.worker.send_statistic.emit({})

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
