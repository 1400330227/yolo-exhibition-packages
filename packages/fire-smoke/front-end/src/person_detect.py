import math
from collections import Counter, deque

import cv2
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from yolo_person.person_utils.sex_detect_worker import detect_gender_and_age
from utils.deep_sort import build_tracker, DeepSort


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # color = compute_color_for_labels(id)
        # label = '{}{:d}'.format("ID:", id)
        label = 'ID:{}'.format(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


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


def get_size_with_pil(label, size=25):
    # 替换成getbox()函数
    font_path = "./configs/simkai.ttf"
    font = ImageFont.truetype(font_path, size, encoding="utf-8")  # simhei.ttf
    text_size = font.getbbox(label)
    return text_size


def put_text_to_cv2_img_with_pil(cv2_img, label, pt, color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8")  # simhei.ttf
    draw.text(pt, label, color, font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式


def bbox_r(width, height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


class PersonDetect():
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
    deepsort = DeepSort("./utils/deep_sort/deep/checkpoint/ckpt.t7",
                        max_dist=0.2, min_confidence=0.3,
                        nms_max_overlap=0.5, max_iou_distance=0.7,
                        max_age=70, n_init=3, nn_budget=100, use_cuda=True)

    def detect_person(self, img, im0s, pred):
        # Process detections
        bbox_xywh = []
        confs = []
        clas = []
        xy = []
        for det in pred:  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det_clone = det.clone()  # Clone the tensor before in-place update
                det_clone[:, :4] = scale_coords(img.shape[2:], det_clone[:, :4], im0s.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det_clone):
                    img_h, img_w, _ = im0s.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = bbox_r(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    # if cls == opt.classes:  # detect classes id
                    if not conf.item() > 0.3:
                        continue
                    bbox_xywh.append(obj)
                    confs.append(conf.item())
                    clas.append(cls.item())
                    xy.append(xyxy)
                    # print('jjjjjjjjjjjjjjjjjjjj', confs)
        return np.array(bbox_xywh), confs, clas, xy

    def get_data(self, img, ori_img, pred, is_reset, isline=False, issex=False):

        """
        ori_img:原始图像
        """
        if is_reset:
            self.deepsort = DeepSort("./utils/deep_sort/deep/checkpoint/ckpt.t7",
                                     max_dist=0.2, min_confidence=0.3,
                                     nms_max_overlap=0.5, max_iou_distance=0.7,
                                     max_age=70, n_init=3, nn_budget=100, use_cuda=True)

        self.idx_frame += 1

        # 将图像进行处理
        bbox_xywh, cls_conf, cls_ids, xy = self.detect_person(img, ori_img, pred)
        outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
        # 根据 choose_line 的值决定是否画黄线
        choose_line = '1' if isline else 0  # 参数 --用户选择是否画黄线 + 用户选择是否识别性别sex_detect
        if choose_line == '1':
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

            if track_id not in self.paths:
                self.paths[track_id] = deque(maxlen=2)
                # self.total_track = track_id
                self.total_track += 1
            self.paths[track_id].append(midpoint)
            previous_midpoint = self.paths[track_id][0]
            origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])

            if line is not None and intersect(midpoint, previous_midpoint, line[0],
                                              line[1]) and track_id not in self.already_counted:
                self.class_counter[self.track_cls] += 1
                self.total_counter += 1
                self.last_track_id = track_id
                # draw red line
                cv2.line(ori_img, line[0], line[1], (0, 0, 255), 10)

                self.already_counted.append(track_id)  # Set already counted for ID to true.

                self.angle = vector_angle(origin_midpoint, origin_previous_midpoint)

                if self.angle > 0:
                    self.up_count += 1
                if self.angle < 0:
                    self.down_count += 1

            if len(self.paths) > 50:
                del self.paths[list(self.paths)[0]]

        # 3. 绘制人员
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            ori_img = draw_boxes(ori_img, bbox_xyxy, identities)

            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

        # 4. 绘制统计信息
        total_label = "客流总数: {}".format(str(int(self.total_track)))  # 参数 --客流总数total_track
        t_size = get_size_with_pil(total_label, 25)
        x1 = 20
        y1 = 50
        # color = compute_color_for_labels(2)
        color = (255, 0, 0)
        cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
        ori_img = put_text_to_cv2_img_with_pil(ori_img, total_label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
        # 统计穿过黄线的人数，只返回，不需要绘图
        # 参数 total_counter up_count down_count
        line_num_label = "穿过黄线人数: {} ({} 向上, {} 向下)".format(str(self.total_counter), str(self.up_count),
                                                                      str(self.down_count))

        line_cross_label = ""
        if self.last_track_id >= 0:  # 参数 -- last_track_id
            line_cross_label = "最新: 行人{}号{}穿过黄线".format(str(self.last_track_id),
                                                                 str("向上") if self.angle >= 0 else str('向下'))

        if isline:
            info = {
                "total": total_label,
                "num_now": f"当前画面人数：{bbox_xywh.shape[0]}",
                "line_cross": line_cross_label,
                "line_num": line_num_label
            }
        else:
            info = {
                "total": total_label,
                "num_now": f"当前画面人数：{bbox_xywh.shape[0]}",
            }
        if issex:
            ori_img = detect_gender_and_age(ori_img)
        # else:
        # print("未选择性别")
        return ori_img, info
