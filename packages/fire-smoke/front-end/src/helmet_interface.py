import sys

import cv2
from IPython.external.qt_for_kernel import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QWidget, \
    QSizePolicy
from qfluentwidgets import CardWidget, BodyLabel, DisplayLabel, TitleLabel, StrongBodyLabel, ComboBox, SubtitleLabel, \
    Slider, PrimaryPushButton, FluentIcon, CheckBox, SingleDirectionScrollArea, ScrollArea, PushButton, SimpleCardWidget
from datasets import load_wights
from setting import names
from utils.ui import removeAllWidgetFromLayout
from i18n.zh_CN import i18n


class HelmetInterface(ScrollArea):
    detect_target_label = '检测人物是否佩戴头盔'

    def __init__(self, text: str, parent=None, worker=None):
        super().__init__(parent=parent)
        # self.label = SubtitleLabel(text, self)
        self.setObjectName('helmet_detect')
        self.worker = worker
        self.weight_paths = load_wights()
        self.all_classes = names
        self.logs = ['设备已经初始化，可以进行目标检测任务']
        # 水平布局，用于放置左右两个 QLabel
        self.layout = QVBoxLayout(self)

        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(True)

        font_h4 = QtGui.QFont()
        font_h4.setFamily("微软雅黑")
        font_h4.setPointSize(12)
        font_h4.setBold(True)

        hbox_video_labels = QHBoxLayout(self)
        label1 = QLabel('安全帽检测')
        label1.setFont(font)
        hbox_video_labels.addWidget(label1)

        self.layout.addLayout(hbox_video_labels)  # 添加标题
        vbox_weight = QVBoxLayout()
        hbox_weight = QHBoxLayout()
        vbox_weight.addLayout(hbox_weight)

        combox_label = QLabel('📁模型选择：')
        combox_label.setFont(font_h4)

        hbox_weight.addWidget(combox_label)
        hbox_weight.addStretch(1)

        # 添加图片检测按钮
        self.image_detect_button = PushButton("💾上传图片检测")
        self.image_detect_button.clicked.connect(self.handler_open_image)
        self.image_detect_button.setEnabled(False)
        # self.image_detect_button.setFixedSize(120, 30)
        hbox_weight.addWidget(self.image_detect_button)

        # 添加视频检测按钮
        self.video_detect_button = PushButton("🎬上传视频检测")
        self.video_detect_button.clicked.connect(self.handler_open_video)
        self.video_detect_button.setEnabled(False)
        # self.video_detect_button.setFixedSize(120, 30)
        hbox_weight.addWidget(self.video_detect_button)

        # 添加显示检测物体按钮
        self.display_objects_button = PushButton("🔍显示检测物体")
        self.display_objects_button.clicked.connect(self.show_detected_objects)
        self.display_objects_button.setEnabled(False)
        # self.display_objects_button.setFixedSize(120, 30)
        hbox_weight.addWidget(self.display_objects_button)

        self.comboBox = ComboBox()
        items = [item for item in self.weight_paths.keys()]
        self.comboBox.addItems(items)
        self.comboBox.currentIndexChanged.connect(lambda index: self.load_model(index, self.comboBox.currentText()))
        vbox_weight.addWidget(self.comboBox)

        self.layout.addLayout(vbox_weight)  # 添加下拉框

        hbox_video = QHBoxLayout()
        self.layout.addLayout(hbox_video)

        # self.layout.setStretchFactor(hbox_video, 3)

        cardWidget1 = SimpleCardWidget()
        hbox_video.addWidget(cardWidget1)
        cardWidget1_vbox = QVBoxLayout(cardWidget1)
        cardWidget1_hbox1 = QHBoxLayout()
        cardWidget1_hbox2 = QVBoxLayout()

        cardWidget1_vbox.addLayout(cardWidget1_hbox1)
        cardWidget1_vbox.addLayout(cardWidget1_hbox2)

        vbox_right1_layout = QVBoxLayout()
        hbox_video.addLayout(vbox_right1_layout)

        cardWidget2 = SimpleCardWidget()
        cardWidget2.setMinimumSize(280, 250)  # 设置大小

        cardWidget3 = SimpleCardWidget()
        cardWidget3.setMinimumSize(280, 250)  # 设置大小
        vbox_right1_layout.addWidget(cardWidget2)
        vbox_right1_layout.addWidget(cardWidget3)
        vbox_right1_layout.setStretchFactor(cardWidget2, 2)
        vbox_right1_layout.setStretchFactor(cardWidget3, 1)

        cardWidget2_vbox = QVBoxLayout(cardWidget2)
        filter_label = QLabel()
        filter_label.setText('过滤条件')
        filter_label.setFixedHeight(20)
        filter_label.setFont(font_h4)
        cardWidget2_vbox.addWidget(filter_label)

        cardWidget2_vbox.setAlignment(Qt.AlignTop)
        cardWidget2_hbox1 = QHBoxLayout(cardWidget2)
        cardWidget2_hbox2 = QHBoxLayout(cardWidget2)
        cardWidget2_hbox3 = QHBoxLayout(cardWidget2)
        cardWidget2_hbox4 = QHBoxLayout(cardWidget2)
        cardWidget2_hbox5 = QHBoxLayout(cardWidget2)

        view = QWidget()
        view.setStyleSheet("QWidget{background: transparent}")
        self.cardWidget2_vbox6 = QVBoxLayout(view)
        self.cardWidget2_vbox6.setAlignment(Qt.AlignTop)

        self.checkbox_list = self.init_checkbox_list(self.all_classes, self.cardWidget2_vbox6)
        for i, checkbox in enumerate(self.checkbox_list):
            checkbox.stateChanged.connect(lambda: self.change_checkbox_state(self.checkbox_list))

        scrollArea = SingleDirectionScrollArea(orient=Qt.Vertical)
        scrollArea.setStyleSheet("QScrollArea{background: transparent; border: none}")
        scrollArea.setWidget(view)

        cardWidget2_vbox.addLayout(cardWidget2_hbox1)
        cardWidget2_vbox.addLayout(cardWidget2_hbox2)
        cardWidget2_vbox.addLayout(cardWidget2_hbox3)
        cardWidget2_vbox.addLayout(cardWidget2_hbox4)
        cardWidget2_vbox.addLayout(cardWidget2_hbox5)
        cardWidget2_vbox.addWidget(scrollArea)

        confidence_threshold_label = QLabel('检测结果可信值：')
        confidence_threshold_label.setFixedHeight(20)
        confidence_threshold_label.setFont(font_h4)
        cardWidget2_hbox1.addWidget(confidence_threshold_label)
        cardWidget2_hbox1.addStretch(1)
        self.confidence_threshold_value_label = SubtitleLabel('30%')
        cardWidget2_hbox1.addWidget(self.confidence_threshold_value_label)

        slider1_value_min = QLabel('0%')
        slider1_value_max = QLabel('100%')
        self.slider1 = Slider(Qt.Horizontal)

        # 设置取值范围和当前值
        self.slider1.setRange(0, 100)
        self.slider1.setValue(30)
        self.slider1.valueChanged.connect(lambda x: self.change_val(x, 'conf'))
        cardWidget2_hbox2.addWidget(slider1_value_min)
        cardWidget2_hbox2.addWidget(self.slider1)
        cardWidget2_hbox2.addWidget(slider1_value_max)

        confidence_iou_label = QLabel('交并比阈值：')
        confidence_iou_label.setFixedHeight(20)
        confidence_iou_label.setFont(font_h4)
        cardWidget2_hbox3.addWidget(confidence_iou_label)
        cardWidget2_hbox3.addStretch(1)
        self.confidence_iou_value_label = SubtitleLabel('70%')
        cardWidget2_hbox3.addWidget(self.confidence_iou_value_label)

        slider2_value_min = QLabel('0%')
        slider2_value_max = QLabel('100%')
        self.slider2 = Slider(Qt.Horizontal)

        # 设置取值范围和当前值
        self.slider2.setRange(0, 100)
        self.slider2.setValue(70)
        self.slider2.valueChanged.connect(lambda x: self.change_val(x, 'iou'))
        cardWidget2_hbox4.addWidget(slider2_value_min)
        cardWidget2_hbox4.addWidget(self.slider2)
        cardWidget2_hbox4.addWidget(slider2_value_max)

        classes_label = QLabel('只检测下列类别：')
        classes_label.setFixedHeight(20)
        classes_label.setFont(font_h4)

        cardWidget2_hbox5.addWidget(classes_label)

        hbox_video.setStretchFactor(cardWidget1, 2)
        hbox_video.setStretchFactor(cardWidget2, 1)

        result_title = StrongBodyLabel(cardWidget1)
        result_title.setText('检测结果')
        result_title.setFixedHeight(20)
        result_title.setFont(font_h4)
        cardWidget1_hbox1.addWidget(result_title)

        cardWidget1_hbox1.addStretch(1)
        label2 = BodyLabel(self.detect_target_label)
        label2.setFont(font_h4)
        label2.setFixedHeight(20)
        cardWidget1_hbox1.addWidget(label2)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(580, 450)  # 设置大小
        self.result_label.setStyleSheet('border:3px solid #009faa;')  # 添加边框并设置背景颜色为黑色
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        cardWidget1_hbox2.addWidget(self.result_label, 1)  # 右侧显示检测后的图像

        action_buttons = QHBoxLayout()
        action_buttons.setAlignment(Qt.AlignCenter)
        # 添加开始检测
        self.detect_button = PrimaryPushButton(FluentIcon.PLAY_SOLID, '开始检测')
        self.detect_button.clicked.connect(self.detect_application)
        self.detect_button.setFixedSize(120, 30)
        action_buttons.addWidget(self.detect_button)

        self.jump_out_button = PushButton(FluentIcon.PAUSE, '停止')
        self.jump_out_button.clicked.connect(self.handler_jump_out)
        self.jump_out_button.setFixedSize(120, 30)

        action_buttons.addWidget(self.jump_out_button)

        cardWidget1_vbox.addLayout(action_buttons)

        cardWidget3_vbox = QVBoxLayout(cardWidget3)
        cardWidget3_vbox.setAlignment(Qt.AlignTop)

        result_details_label = QLabel()
        result_details_label.setText('详细信息')
        result_details_label.setFixedHeight(20)
        result_details_label.setFont(font_h4)
        cardWidget3_vbox.addWidget(result_details_label)

        self.resultWidget = QListWidget(cardWidget3)
        self.resultWidget.setStyleSheet(
            "QListWidget{background-color: rgba(12, 28, 77, 0);border-radius:0px;font-size: 16px;}")
        cardWidget3_vbox.addWidget(self.resultWidget)
        # 添加退出按钮
        # self.exit_button = QPushButton("❌退出")
        # self.exit_button.clicked.connect(self.exit_application)
        # self.exit_button.setFixedSize(120, 30)

        # cardWidget4 = QWidget()
        # cardWidget4.setStyleSheet("QWidget{background: transparent}")
        # self.cardWidget4_vbox1 = QVBoxLayout(cardWidget4)
        # self.cardWidget4_vbox1.setAlignment(Qt.AlignTop)

        self.logs_widget = QListWidget()
        self.logs_widget.setStyleSheet("QListWidget{background: transparent; border: none}")
        self.logs_widget.addItems(self.logs)
        self.logs_widget.resize(20000, 50)
        self.logs_widget.setMaximumHeight(50)

        # scrollArea1 = SingleDirectionScrollArea(orient=Qt.Vertical)
        # scrollArea1.setStyleSheet("QScrollArea{background: transparent; border: none}")
        # scrollArea1.setWidget(self.logs_widget)
        # scrollArea1.setMaximumHeight(50)
        self.layout.addWidget(self.logs_widget)

        self.init_widget()
        self.add_event_listener()
        self.init_model('helmet.pt')

    def init_model(self, model_path):
        index = list(self.weight_paths.keys()).index(model_path)
        if index != -1:
            self.comboBox.setCurrentIndex(index)
            self.load_model(index, model_path)

    def add_event_listener(self):
        self.worker.send_img.connect(lambda x: self.show_image(x, self.result_label))
        self.worker.send_raw.connect(lambda x: self.show_image(x, self.result_label))
        self.worker.send_statistic.connect(lambda x: self.show_statistic(x, self.resultWidget))

    def set_logs(self, logs):
        if isinstance(logs, list):
            self.logs = self.logs + logs
        else:
            self.logs.append(logs)
        self.logs_widget.clear()
        self.logs_widget.addItems(self.logs)

    @staticmethod
    def init_checkbox_list(all_classes, layout):
        removeAllWidgetFromLayout(layout)
        checkbox_list = []
        for k, v in all_classes.items():
            if k > 5:
                continue
            checkbox = CheckBox(i18n(str(v)))
            checkbox.setTristate(True)
            checkbox.setFixedHeight(20)
            checkbox.resize(200, 20)
            checkbox.setCheckState(Qt.Unchecked)
            checkbox_list.append(checkbox)
            layout.addWidget(checkbox)
        return checkbox_list

    @staticmethod
    def show_image(img_src, label):
        if img_src.size == 0:
            label.clear()
            return
        try:
            frame = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)

            label.setPixmap(QPixmap.fromImage(img))
            label.setScaledContents(True)

        except Exception as e:
            print(repr(e))

    @staticmethod
    def show_statistic(statistic_dic, label):
        try:
            label.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            label.addItems(results)

        except Exception as e:
            print(repr(e))

    def init_widget(self):
        self.layout.setContentsMargins(20, 18, 20, 20)
        self.setStyleSheet("QScrollArea{background: transparent; border: none}")

    def change_val(self, value, flag):
        if flag == 'conf':
            self.slider1.setValue(value)
            self.worker.conf = round(value / 100, 1)
            self.confidence_threshold_value_label.setText(str(value) + "%")
        elif flag == 'iou':
            self.slider2.setValue(value)
            self.worker.iou = round(value / 100, 1)
            self.confidence_iou_value_label.setText(str(value) + "%")

    def change_checkbox_state(self, checkbox_list):
        check_list, _ = self.get_checkbox(checkbox_list)
        self.worker.set_classes(check_list)

    @staticmethod
    def get_checkbox(checkbox_list):
        k = 0
        l_chk = checkbox_list
        check_list = []
        for i, checkbox in enumerate(l_chk):
            if checkbox.isChecked():  # isChecked()判断复选框是否被选中
                check_list.append(i)
            k = k + 1
        return check_list, k

    def load_model(self, key, value):
        if key == -1:
            return
        model_path = self.weight_paths[value]
        self.worker.set_model_path(model_path)
        self.all_classes = self.worker.get_classes()

        self.checkbox_list = self.init_checkbox_list(self.all_classes, self.cardWidget2_vbox6)
        for i, checkbox in enumerate(self.checkbox_list):
            checkbox.stateChanged.connect(lambda: self.change_checkbox_state(self.checkbox_list))

        self.image_detect_button.setEnabled(True)
        self.video_detect_button.setEnabled(True)
        self.display_objects_button.setEnabled(True)

    def handler_open_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)")
        if not image_path:
            print('沒有找到文件')
        self.worker.set_source(image_path)
        self.worker.frame_show_1(image_path)
        log = '您已经打开文件：' + image_path
        self.set_logs([log])

    def open_file(self, args):
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", args)
        if not image_path:
            print('沒有找到文件')
        self.worker.set_source(image_path)

    def handler_open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, 'Video/image', '', "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                             "*.jpg *.png)")
        if not video_path:
            print('沒有找到文件')
        self.worker.set_source(video_path)
        self.worker.frame_show_1(video_path)
        log = '您已经打开文件：' + video_path
        self.set_logs([log])

    def show_detected_objects(self):
        if self.current_results:
            print(self.current_results)

    def detect_application(self):
        self.worker.jump_out = False
        if not self.worker.isRunning():
            self.worker.start()

    def handler_jump_out(self):
        self.worker.jump_out = True

    def exit_application(self):
        # 终止程序运行
        sys.exit()
