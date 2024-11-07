def read_path(n):
    with open('path.txt', 'r') as file:
        lines = file.readlines()
        return lines[n-1].split(' ')[1].strip()
def update_path(path, n):
    try:
        with open('path.txt', 'r') as file:
            lines = file.readlines()
        if n == 1:
            lines[0] = f"input_video_path {path}\n"
        elif n == 2:
            lines[1] = f"output_video_path {path}\n"
        else:
            return False
        with open('path.txt', 'w') as file:
            file.writelines(lines)
        return True
    except:
        return False
def update_redline_sex_video(i,n):
    try:
        with open('path.txt', 'r') as file:
            lines = file.readlines()
        if n==3:
            lines[2] = f"redline {i}\n"
        elif n==4:
            lines[3] = f"default_video {i}\n"
        else:
            return False
        with open('path.txt', 'w') as file:
            file.writelines(lines)
        return True
    except:
        return False
