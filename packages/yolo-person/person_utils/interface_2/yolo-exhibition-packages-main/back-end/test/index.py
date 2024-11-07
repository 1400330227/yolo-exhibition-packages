import cv2
import numpy as np

# 创建一个黑色图像
img = np.zeros((400, 400, 3), dtype=np.uint8)

# 在图像上绘制白色文本
text = "Hello, OpenCV"
org = (50, 200)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 255, 255)  # BGR格式，白色
thickness = 1

if __name__ == '__main__':
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness)

    # 显示图像
    cv2.imshow('Text', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
