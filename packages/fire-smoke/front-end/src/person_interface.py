import sys

import cv2
from IPython.external.qt_for_kernel import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QWidget, \
    QSizePolicy
from qfluentwidgets import CardWidget, BodyLabel, DisplayLabel, TitleLabel, StrongBodyLabel, ComboBox, SubtitleLabel, \
    Slider, PrimaryPushButton, FluentIcon, CheckBox, SingleDirectionScrollArea, ScrollArea, PushButton, \
    SimpleCardWidget, HeaderCardWidget
from datasets import load_wights
from setting import names
from utils.ui import removeAllWidgetFromLayout
from i18n.zh_CN import i18n


class PersonInterface(ScrollArea):
    def __init__(self, text: str, subtext='检测图像是否存在火焰烟雾', parent=None, worker=None):
        super(PersonInterface, self).__init__(parent)
        self.text = text
        self.subtext = subtext
        self.worker = worker
        self.view = QWidget(self)
        self.layout = QVBoxLayout(self.view)  # 主要布局

        title = QLabel(text)
        title.setStyleSheet("color: #332b1f;")
        row1_Widget = QWidget(self.view)
        row1_Widget.setStyleSheet("background-color: #e0ddd4;")
        row1_h_layout = QHBoxLayout(row1_Widget)
        row1_h_layout.addWidget(title)

        self.layout.addWidget(row1_Widget)

        self.init_widget()

    def init_widget(self):
        self.setObjectName('PersonInterface')
        self.setWidget(self.view)
        self.resize(778, 778)
        self.setStyleSheet("QScrollArea{background: transparent; border: none;}")
