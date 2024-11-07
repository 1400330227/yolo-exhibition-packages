import torch
from ultralytics import YOLO
import os
import cv2
from collections import defaultdict
import numpy as np
import subprocess

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
project_root_path = os.path.dirname(current_dir_path)


# weight = os.path.join('weights', 'best.pt')
weight = os.path.join('weights', 'last.pt')
# weight = os.path.join('../weights', 'last.pt')

class Detector(object):
    def __init__(self):
        self.model = None
        self.img_size = 640
        self.threshold = 0.4
        self.max_frame = 160
        # self.weights = project_root_path + '/weights/best.pt'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.init_model(self.weights)

    def init_model(self, weight):
        _model = YOLO(weight)
        _model = _model.to(self.device)
        return _model

    def detect_images(self, images):
        model = self.init_model(weight)
        results = model.predict(images, conf=0.5)
        return results

    def detect_video(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20.0  # 如果无法获取帧率，则设置为默认值
        # save_path = '../uploads/videos/labels/' + filename
        fourcc = cv2.VideoWriter.fourcc(*'VP80')
        out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

        model = self.init_model(weight)
        # model = self.model

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, save=False, conf=0.5)
                annotated_frame = results[0].plot()
                # cv2.imshow("YOLO11 Tracking", annotated_frame)
                out.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = Detector()
    result = detector.detect_images('../uploads/images/trains/000001.jpg')
    result[0].show()
    # model = detector.init_model(project_root_path + '/weights/best.pt')
    # detector.detect_video('../datasets/videos/test.mp4', name='test.mp4')
    # metrics = model.val()
    # detector.detect_video("https://youtu.be/LNwODJXcvt4")
