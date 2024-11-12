import cv2


def succeed_cap(index):
    # 获取摄像头对
    cap = cv2.VideoCapture(index)
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        if not ret:
            print("无法捕获帧")
            break
        # ret和frame的type
        print("ret.type", type(ret))  # bool
        print("frame.type", type(frame))  # numpy(array)
        print("frame.shape---", frame.shape)
        print("frame-----", frame)
        # 显示帧
        cv2.imshow('摄像头画面', frame)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) == ord('q'):
            break
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    index = 0
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"找到摄像头，索引为 {index}")
            succeed_cap(index)
        index += 1
