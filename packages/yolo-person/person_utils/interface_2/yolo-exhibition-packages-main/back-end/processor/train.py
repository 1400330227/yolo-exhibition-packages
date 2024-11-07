from ultralytics import YOLO
import os

current_file_path = os.path.abspath(__file__)


if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # Load an official Detect model
    results = model.track("../datasets/videos/test.mp4", show=True)  # Tracking with default tracker
    current_file_path = os.path.abspath(__file__)
    dirname = os.path.dirname(current_file_path)
    print(current_file_path)
    print(dirname)