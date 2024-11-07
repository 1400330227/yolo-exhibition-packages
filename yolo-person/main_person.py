import subprocess
from person_utils.sex_detect import detect_gender_and_age
person_detection = 0
sex_dection = 0
if person_detection :
    result = subprocess.run(["D:\software\py_cun\yolov10\pythonProject\Scripts\python.exe", "main.py"], cwd="./person_utils")
    if sex_dection :
        video_path = './person_utils/output_video.mp4'
        output_path = './person_utils/output_sex_video.mp4'
        print("执行sex_detection.py 79行，使用默认路径")
        detect_gender_and_age(video_path, output_path)
