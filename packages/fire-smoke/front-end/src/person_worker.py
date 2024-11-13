import time

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
import os
import utils.draw_boxes as Boxes
import utils.i18n as Statistics
from person_detect import PersonDetect, letterbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(suppress=True)

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes


class PersonWorker(QThread):
    send_img = pyqtSignal(np.ndarray)  # 检测结果图像
    send_raw = pyqtSignal(np.ndarray)  # 检测源图像
    send_statistic = pyqtSignal(dict)  # 检测结果统计信息

    send_msg = pyqtSignal(str)  # 日志
    send_fps = pyqtSignal(str)  # 帧率

    def __init__(self):
        super(PersonWorker, self).__init__()
        self.source = None
        self.model_path = None
        self.model = None
        self.conf = None
        self.iou = None
        self.classes = []

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
        images = cv2.imread(images)
        # names = model.module.names if hasattr(model, 'module') else model.names  # 分类信息
        detector = PersonDetect()
        try:
            conf = self.conf if self.conf else 0.3
            iou = self.iou if self.iou else 0.7
            # classes = self.classes if self.classes else None
            # results = model.predict(images, conf=conf, iou=iou, classes=classes, stream=True)
            img = letterbox(images)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(torch.device('cuda'))
            img_tensor = img_tensor.half()
            img_tensor /= 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # results = model.predict(frame, conf=conf, iou=iou, classes=[0])[0]  # 检测
            results = model(img_tensor, conf=conf, iou=iou, classes=[0], verbose=False)  # 检测人,无需其他类别信息
            pred = results[0].boxes.data

            img, classes = detector.get_data(img_tensor, images, [pred])
            results = model(images, conf=conf, iou=iou, classes=[0], verbose=False)
            self.send_img.emit(results[0].plot())
            self.send_statistic.emit(classes)
        except Exception as e:
            print(e)
        # return results

    def detect_video(self, video_path, save_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            if fps == 0:
                fps = 20.0  # 如果无法获取帧率，则设置为默认值

            model = self.init_model(self.model_path)
            detector = PersonDetect()
            # names = model.module.names if hasattr(model, 'module') else model.names  # 分类信息

            len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame = 0
            average_fps = 0.0
            is_reset = False
            while cap.isOpened():
                if self.jump_out:
                    break

                if video_frame == len_frames:
                    video_frame = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    is_reset = True

                success, frame = cap.read()
                print('视频帧获取是否成功：{}'.format(success))
                if success:
                    start = time.time()

                    conf = self.conf if self.conf else 0.3
                    iou = self.iou if self.iou else 0.7
                    # classes = self.classes if self.classes else None

                    # 对图像进行处理以适应后续操作
                    img = letterbox(frame)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img_tensor = torch.from_numpy(img).to(torch.device('cuda'))
                    img_tensor = img_tensor.half()
                    img_tensor /= 255.0
                    if img_tensor.ndimension() == 3:
                        img_tensor = img_tensor.unsqueeze(0)

                    # results = model.predict(frame, conf=conf, iou=iou, classes=[0])[0]  # 检测
                    results = model(img_tensor, conf=conf, iou=iou, classes=[0], verbose=False)  # 检测人,无需其他类别信息
                    pred = results[0].boxes.data

                    # 判断是否检测性别或者向上向下
                    isline = False
                    issex = False
                    if self.classes is not None:
                        if 0 in self.classes:
                            isline = True
                        if 1 in self.classes:
                            issex = True
                    # 将图像，识别结果传入处理函数，获取到画框后的图像以及统计信息
                    img, classes = detector.get_data(img_tensor, frame, [pred], is_reset, isline=isline, issex=issex)
                    is_reset = False

                    end = time.time()
                    video_frame = video_frame + 1
                    video_fps = 1.0 / (end - start)
                    average_fps += video_fps

                    print('当前帧：{}'.format(video_frame))
                    print("平均帧率: %.1f" % (average_fps / video_frame))

                    # 休眠一段时间，以控制帧率
                    # if video_fps > fps:
                    #     time.sleep((1 / fps - (end - start)))
                    # Boxes.draw_boxes(frame, results)
                    # classes = Statistics.statistics_classes(results, names)
                    # annotation_frame = results[0].plot(font_size=10)

                    self.send_img.emit(
                        img if isinstance(img, np.ndarray) else img[0])
                    # self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
                    self.send_statistic.emit(classes)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    print('结束了')
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(e)
        finally:
            del detector

    def set_source(self, source):
        self.source = source

    def set_model_path(self, model_path):
        self.model_path = model_path if model_path else None

    def set_classes(self, classes):
        self.classes = classes if len(classes) > 0 else None

    def frame_show_1(self, video_path):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        if success:
            print('视频第一帧获取成功：{}')
            self.send_img.emit(frame if isinstance(frame, np.ndarray) else frame[0])
        cap.release()
        return True
