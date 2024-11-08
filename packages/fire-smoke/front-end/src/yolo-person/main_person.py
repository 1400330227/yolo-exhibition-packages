import subprocess
person_detection = 1
sex_dection = 0
if person_detection == 1 :
    result = subprocess.run(["D:\software\py_cun\yolov10\pythonProject\Scripts\python.exe", "main.py"], cwd="./person_utils")


