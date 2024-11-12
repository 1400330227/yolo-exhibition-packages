import time

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from translate import Translator
from ultralytics import YOLO
import os
import utils.draw_boxes as Boxes
# import utils.i18n as Statistics
import utils.statistics_classes as Statistics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(suppress=True)

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes


class Worker(QThread):
    send_img = pyqtSignal(np.ndarray)  # 检测结果图像
    send_raw = pyqtSignal(np.ndarray)  # 检测源图像
    send_statistic = pyqtSignal(dict)  # 检测结果统计信息

    send_msg = pyqtSignal(str)  # 日志
    send_fps = pyqtSignal(str)  # 帧率
    translator = Translator(from_lang="en", to_lang="zh")

    def __init__(self):
        super(Worker, self).__init__()
        self.source = None
        self.model_path = None
        self.model = None
        self.conf = None
        self.iou = None
        self.classes = None

        self.jump_out = False  # 停止运行
        self.is_continue = True

    def run(self):
        try:
            if self.source is None:
                print('请上传文件')
                return
            if self.source.split('.')[-1].lower() in IMG_FORMATS:
                self.detect_image(self.source)
            elif self.source.split('.')[-1].lower() in VID_FORMATS:
                self.detect_video(video_path=self.source, save_path='')
        except Exception as e:
            self.send_msg.emit('%s' % e)

    def load_model(self, model_path):
        if model_path:
            self.model_path = model_path

    def get_classes(self):
        if self.model_path is not None:
            model = self.init_model(self.model_path)
            names = model.module.names if hasattr(model, 'module') else model.names  # 分类信息
            del model
            return names
        else:
            return []

    def init_model(self, model_path):
        if model_path:
            model = YOLO(model_path)
            self.model = model.to(device)
            return model
        return False

    def detect_image(self, images):
        model = self.init_model(self.model_path)
        names = model.module.names if hasattr(model, 'module') else model.names  # 分类信息
        try:
            conf = self.conf if self.conf else 0.3
            iou = self.iou if self.iou else 0.7
            classes = self.classes if self.classes else None

            results = model.predict(images, conf=conf, iou=iou, classes=classes, stream=True)
            for i, result in enumerate(results):
                frame = result.plot()
                classes = Statistics.statistics_classes(results, names)

                self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
                self.send_statistic.emit(classes)
                # if isinstance(images, list):
                #     Boxes.draw_boxes(frame, results)
                #     classes = Statistics.statistics_classes(results, names)
                #
                #     self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
                #     self.send_statistic.emit(classes)
                # elif isinstance(images, str): # 单张照片
                #     image = cv2.imread(images)
        except Exception as e:
            print(e)
        # return results

    def detect_video(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        if fps == 0:
            fps = 20.0  # 如果无法获取帧率，则设置为默认值

        model = self.init_model(self.model_path)
        names = model.module.names if hasattr(model, 'module') else model.names  # 分类信息

        video_frame = 0
        average_fps = 0.0
        try:
            while True:
                if self.jump_out:
                    break

                success, frame = cap.read()
                print('视频帧获取是否成功：{}'.format(success))
                if success:
                    start = time.time()

                    conf = self.conf if self.conf else 0.3
                    iou = self.iou if self.iou else 0.7
                    classes = self.classes if self.classes else None

                    results = model.predict(frame, conf=conf, iou=iou, classes=classes)  # 检测

                    end = time.time()
                    video_frame = video_frame + 1
                    video_fps = 1.0 / ((end - start) + 0.001)
                    average_fps += video_fps

                    print('当前帧：{}'.format(video_frame))
                    print("平均帧率: %.1f" % (average_fps / video_frame))

                    # Boxes.draw_boxes(frame, results)
                    classes = Statistics.statistics_classes(results, names)
                    annotation_frame = results[0].plot(font_size=10)

                    self.send_img.emit(
                        annotation_frame if isinstance(annotation_frame, np.ndarray) else annotation_frame[0])
                    # self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
                    self.send_statistic.emit(classes)
                    # 休眠一段时间，以控制帧率
                    if video_fps > fps:
                        time.sleep(1 / fps - (end - start))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    print('结束了')
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)

    def set_source(self, source):
        self.source = source

    def set_model_path(self, model_path):
        self.model_path = model_path if model_path else None

    def set_classes(self, classes):
        self.classes = classes if len(classes) > 0 else None

    def frame_show_1(self, video_path):
        # print(video_path)
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if success:
            # print('视频第一帧获取成功：{}')
            self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
        cap.release()
        return True
