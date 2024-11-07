import subprocess
from person_utils.sex_detect import detect_gender_and_age
person_detection = 1
sex_dection = 0
if person_detection == 1 :
    result = subprocess.run(["D:\software\py_cun\yolov10\pythonProject\Scripts\python.exe", "main.py"], cwd="./person_utils")
    if sex_dection == 1 :
        video_path = './person_utils/output_video.mp4'
        output_path = './person_utils/output_sex_video.mp4'
        detect_gender_and_age(video_path, output_path)


