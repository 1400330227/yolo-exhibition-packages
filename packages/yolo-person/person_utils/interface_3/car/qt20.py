# 方案二
from PySide6 import QtWidgets, QtCore, QtGui
import cv2 as cv
import time
from threading import Thread
from ultralytics import YOLOv10
import sys

class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

        # 设置按键与其后端对应的功能
        self.camBtn.clicked.connect(self.startCamera)#摄像头
        self.pauseBtn.clicked.connect(self.pause)# 暂停
        self.stopBtn.clicked.connect(self.stop)# 关闭
        self.pictureBtn.clicked.connect(self.startPicture)#图片检测
        self.videoBtn.clicked.connect(self.startVideo)#视频检测

        # 线程，用于处理摄像头和视频的帧
        self.frameToAnalyze = []
        self.frameToAnalyzeVideo = []
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()
        Thread(target=self.frameAnalyzeThreadFuncVideo, daemon=True).start()

        # 定时器，用于定期处理摄像头和视频的帧
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.show_video)

        self.model = None
        self.loadModel("模型1")

        # 初始化检测结果表格
        self.initTable()

        # 暂停
        self.video_paused = False

    def setupUI(self):

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QGridLayout(centralWidget)

        # 左上部分（视频显示）
        self.label_top_left = QtWidgets.QLabel("图像显示")
        self.label_top_left = QtWidgets.QLabel(self)
        
        self.label_top_left.setMinimumSize(800, 600)
        self.label_top_left.setScaledContents(True)
        mainLayout.addWidget(self.label_top_left, 0, 0)

        # 右上部分（操作选项）
        self.groupBox_top_right = QtWidgets.QGroupBox("操作选项")
        buttonsLayout = QtWidgets.QVBoxLayout(self.groupBox_top_right)
        self.modelComboBox = QtWidgets.QComboBox()
        self.modelComboBox.addItem("模型1")
        self.modelComboBox.currentTextChanged.connect(self.updateModel)
        buttonsLayout.addWidget(self.modelComboBox)
        
        self.pictureBtn = QtWidgets.QPushButton('🖼️图片文件', self.groupBox_top_right)# UI
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件', self.groupBox_top_right)
        self.camBtn = QtWidgets.QPushButton('📹摄像头', self.groupBox_top_right)

        pauseStopLayout = QtWidgets.QHBoxLayout()

        self.pauseBtn = QtWidgets.QPushButton('⏸️暂停', self.groupBox_top_right)#ui
        self.stopBtn = QtWidgets.QPushButton('🛑停止', self.groupBox_top_right)
        pauseStopLayout.addWidget(self.pauseBtn)
        pauseStopLayout.addWidget(self.stopBtn)

        buttonsLayout.addWidget(self.pictureBtn)
        buttonsLayout.addWidget(self.videoBtn)
        buttonsLayout.addWidget(self.camBtn)
        buttonsLayout.addLayout(pauseStopLayout)

        mainLayout.addWidget(self.groupBox_top_right, 0, 1)

        # 左下部分（结果显示表格）
        self.resultTable = QtWidgets.QTableWidget(0, 2)
        self.resultTable.setMaximumSize(800, 200)
        self.resultTable.setHorizontalHeaderLabels(['检测种类', '数量'])
        mainLayout.addWidget(self.resultTable, 1, 0)

        # 右下部分（颜色选项）
        self.colorOptionsGroup = QtWidgets.QGroupBox("颜色选项")
        self.colorOptionsGroup.setMinimumSize(200, 100)

        self.top = QtWidgets.QGroupBox("颜色")
        self.redCheckBox = QtWidgets.QRadioButton("红")
        self.blackCheckBox = QtWidgets.QRadioButton("黑")
        self.whiteCheckBox = QtWidgets.QRadioButton("白")
        self.blueCheckBox = QtWidgets.QRadioButton("蓝")
        self.allCheckBox = QtWidgets.QRadioButton("全部")
        self.redCheckBox.toggled.connect(lambda checked: self.updateModel("模型1" if checked else "其他模型"))#勾选Checkbox
        self.blackCheckBox.toggled.connect(lambda checked: self.updateModel("模型1" if checked else "其他模型"))
        self.whiteCheckBox.toggled.connect(lambda checked: self.updateModel("模型1" if checked else "其他模型"))
        self.blueCheckBox.toggled.connect(lambda checked: self.updateModel("模型1" if checked else "其他模型"))
        self.allCheckBox.toggled.connect(lambda checked: self.updateModel("模型1" if checked else "其他模型"))
        
        self.allCheckBox.setChecked(True)

        topLayout = QtWidgets.QVBoxLayout()
        topLayout.addWidget(self.allCheckBox)
        topLayout.addWidget(self.redCheckBox)
        topLayout.addWidget(self.blackCheckBox)
        topLayout.addWidget(self.whiteCheckBox)
        topLayout.addWidget(self.blueCheckBox)
        self.top.setLayout(topLayout)

        colorOptionsLayout = QtWidgets.QVBoxLayout()
        colorOptionsLayout.addWidget(self.top)
        self.colorOptionsGroup.setLayout(colorOptionsLayout)

        mainLayout.addWidget(self.colorOptionsGroup, 1, 1)

        mainLayout.setSpacing(10)
        centralWidget.setLayout(mainLayout)
        

    def initTable(self):
        self.resultTable.setColumnWidth(0, 150)
        self.resultTable.setColumnWidth(1, 100)

    def loadModel(self,modelName):
        if modelName=="模型1":
            if self.allCheckBox.isChecked():
                self.model = YOLOv10('D:\\.yolo\\yolov10\\yolov10-main\\runs\\detect\\train30\\weights\\best.pt')
            elif self.blackCheckBox.isChecked():
                self.model=YOLOv10('D:\\.yolo\\yolov10\\yolov10-main\\runs\\detect\\train50\\weights\\best.pt')
            elif self.whiteCheckBox.isChecked():
                self.model=YOLOv10('D:\\.yolo\\yolov10\\yolov10-main\\runs\\detect\\train59\\weights\\best.pt')
            elif self.redCheckBox.isChecked():
                self.model=YOLOv10('D:\\.yolo\\yolov10\\yolov10-main\\runs\\detect\\train60\\weights\\best.pt')


    def updateModel(self, modelName):
        self.loadModel(modelName)

    def  startCamera(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            return
        if not self.timer_camera.isActive():
            self.timer_camera.start(50)

    def show_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frameToAnalyze.append(frame)

    def analyze_frame(self, frame):
        results = self.model(frame)[0]
        img = results.plot(line_width=3)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)

        label_size = self.label_top_left.size()
        pixmap = QtGui.QPixmap.fromImage(qImage).scaled(label_size, QtCore.Qt.KeepAspectRatio)
        QtCore.QMetaObject.invokeMethod(self.label_top_left, "setPixmap", QtCore.Q_ARG(QtGui.QPixmap, pixmap))

# qt16.py照这个改
        detected_objects = results.boxes
        class_names = results.names

        self.resultTable.setRowCount(0)

        class_counts = {}
        for box in detected_objects:
            class_id = int(box.cls)
            class_name = class_names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            row_position = self.resultTable.rowCount()
            self.resultTable.insertRow(row_position)
            self.resultTable.setItem(row_position, 0, QtWidgets.QTableWidgetItem(class_name))
            self.resultTable.setItem(row_position, 1, QtWidgets.QTableWidgetItem(str(count)))

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue
            frame = self.frameToAnalyze.pop(0)
            self.analyze_frame(frame)

    def startVideo(self):
        options = QtWidgets.QFileDialog.Options()
        videosource, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.wmv)", options=options)
        if videosource:
            self.cap = cv.VideoCapture(videosource)
            if not self.cap.isOpened():
                return
            if not self.timer_video.isActive():
                self.timer_video.start(50)

    def show_video(self):
        if self.video_paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer_video.stop()
            return
        self.frameToAnalyzeVideo.append(frame)

    def frameAnalyzeThreadFuncVideo(self):
        while True:
            if not self.frameToAnalyzeVideo:
                time.sleep(0.01)
                continue
            frame = self.frameToAnalyzeVideo.pop(0)
            self.analyze_frame(frame)

    def startPicture(self):
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.png *.bmp)", options=options)
        if filename:
            frame = cv.imread(filename)
            self.analyze_frame(frame)

    def pause(self):
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.timer_video.stop()
            self.pauseBtn.setText("▶️继续")
        else:
            self.timer_video.start(50)
            self.pauseBtn.setText("⏸️暂停")

    def stop(self):
        self.video_paused = False
        self.timer_camera.stop()
        self.timer_video.stop()
        if hasattr(self, 'cap'):
            self.cap.release()

app = QtWidgets.QApplication([])
window = MWindow()
window.show()
sys.exit(app.exec())
