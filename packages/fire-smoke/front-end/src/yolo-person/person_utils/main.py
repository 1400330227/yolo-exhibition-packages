import cv2
import torch
import warnings
import argparse
import numpy as np

from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes
from utils.general import check_img_size
from utils.person_detect_yolov10 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
# count
from collections import Counter,deque
import math
from PIL import Image, ImageDraw, ImageFont

def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def get_size_with_pil(label,size=25):
    # 替换成getbox()函数
    font_path = "./configs/simkai.ttf"
    font = ImageFont.truetype(font_path, size, encoding="utf-8")  # simhei.ttf
    text_size = font.getbbox(label)
    return text_size
'''
    # 此处因为pillow版本问题，最新的pillow版本删去了getsize()函数。->pip install Pillow==9.5//替换成getbbox()函数
    font_path = "./configs/simkai.ttf"
    font = ImageFont.truetype(font_path, size, encoding="utf-8")  # simhei.ttf
    text_size = font.getsize(label)
    return text_size
'''


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式


colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ]);

def get_color(c, x, max):
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

def compute_color_for_labels(class_id,class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))

class yolo_reid():

    def __init__(self, cfg, args, path, weights):
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
        use_cuda = args.use_cuda and torch.cuda.is_available() # cpu相关
        if not use_cuda:
            warnings.warn("哥们，你全在用cpu跑！这样很慢", UserWarning) # cpu相关
        else :
            warnings.warn("哥们不错呦，开始调用GPU...", UserWarning)
        '''
        self.person_detect = Person_detect(self.args, self.video_path)
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size
        self.dataset = LoadImages(self.video_path, img_size=imgsz)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)
        '''
        self.weights = weights
        self.person_detect = Person_detect(self.args, self.weights, self.video_path)
        self.vid_cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
        output_path = './person_output.mp4' # 默认
        self.out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.dataset = LoadImages(self.video_path, img_size=check_img_size(args.img_size, s=32))
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)

    def deep_sort(self):
        idx_frame = 0
        results = []
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        up_count = 0
        down_count = 0
        class_counter = Counter()  # store counts of each detected class
        already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs

        for video_path, img, ori_img, vid_cap in self.dataset:
            idx_frame += 1
            t1 = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 根据 choose_line 的值决定是否画黄线
            choose_line = 0 # 参数 --用户选择是否画黄线 + 用户选择是否识别性别sex_detect
            if choose_line == '1' :
                # 1.视频中间画行黄线
                line = [(0, int(0.48 * ori_img.shape[0])), (int(ori_img.shape[1]), int(0.48 * ori_img.shape[0]))]
                cv2.line(ori_img, line[0], line[1], (0, 255, 255), 4)
            else:

                line = None

            # 2. 统计人数
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint = tlbr_midpoint(bbox)
                origin_midpoint = (
                midpoint[0], ori_img.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                if track_id not in paths:
                    paths[track_id] = deque(maxlen=2)
                    total_track = track_id
                paths[track_id].append(midpoint)
                previous_midpoint = paths[track_id][0]
                origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])

                if line is not None and intersect(midpoint, previous_midpoint, line[0], line[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id
                    # draw red line
                    cv2.line(ori_img, line[0], line[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)

                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1
                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # 3. 绘制人员
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

            # 4. 绘制统计信息
            label = "客流总数: {}".format(str(total_track)) # 参数 --客流总数total_track
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 50
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))
            # 参数 total_counter up_count down_count
            label = "穿过黄线人数: {} ({} 向上, {} 向下)".format(str(total_counter), str(up_count), str(down_count))
            t_size = get_size_with_pil(label, 25)
            x1 = 20
            y1 = 100
            color = compute_color_for_labels(2)
            cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

            if last_track_id >= 0: # 参数 -- last_track_id
                label = "最新: 行人{}号{}穿过黄线".format(str(last_track_id),
                                                          str("向上") if angle >= 0 else str('向下'))
                t_size = get_size_with_pil(label, 25)
                x1 = 20
                y1 = 150
                color = compute_color_for_labels(2)
                cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
                ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

            end = time_synchronized()

            # 将处理后的帧写入视频文件
            self.out.write(ori_img)

            # 实时显示输出结果
            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # 参数 bbox_xywh.shape[0]
            # 实时显示运行过程中的相关信息
            self.logger.info("{}/time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(idx_frame, end - t1, 1 / (end - t1),
                                     bbox_xywh.shape[0], len(outputs)))
        # 释放资源
        self.vid_cap.release()
        self.out.release()
        '''
        # 在结果识别完后，使用单独的窗口展示结果
        output_path = './output_video.mp4'
        print("第254行", output_path)
        cap = cv2.VideoCapture(output_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Output Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        '''


def parse_args():
    parser = argparse.ArgumentParser()
    # 默认
    input_video_path = './input_video.mp4'
    parser.add_argument("--video_path", default=input_video_path, type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') #cpu相关
    # yolov10
    parser.add_argument('--weights', nargs='+', type=str, default='./models/yolov10s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True) #cpu相关

    return parser.parse_args()


# if __name__ == '__main__':
#     args = parse_args()
#     cfg = get_config()
#     cfg.merge_from_file(args.config_deepsort)
#     yolo_reid = yolo_reid(cfg, args, path=args.video_path)
#     with torch.no_grad():
#         yolo_reid.deep_sort()

def main_person():
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    yolo_reid_instance = yolo_reid(cfg, args, path=args.video_path, weights='./models/yolov10s.pt')
    with torch.no_grad():
        yolo_reid_instance.deep_sort()