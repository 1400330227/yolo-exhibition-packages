import os

from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, jsonify, flash, send_from_directory, send_file, Response
from processor.Detector import Detector
import json
import os
import pandas as pd
from pathlib import Path
import re

# 当前文件路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 当前项目根路径
PROJECT_ROOT_PATH = os.path.dirname(CURRENT_FILE_PATH)
# 上传文件路径
UPLOAD_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, 'uploads')
# 训练文件路径
IMAGES_TRAINS_PATH = os.path.join('uploads', 'images', 'trains')
# 检测文件路径
IMAGES_LABELS_PATH = os.path.join('uploads', 'images', 'labels')

VIDEOS_TRAINS_PATH = os.path.join('uploads', 'videos', 'trains')
VIDEOS_LABELS_PATH = os.path.join('uploads', 'videos', 'labels')

ALLOWED_EXTENSIONS_IMAGES = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS_VIDEOS = set(['mp4', 'gif'])

print(PROJECT_ROOT_PATH, 'project_root_path')
print(UPLOAD_FOLDER_PATH, 'UPLOAD_FOLDER')

host = '127.0.0.1'
port = 5003
app = None

detect_results = {}


def allowed_file(filename):
    return '.' in filename and (allowed_video(filename) or allow_image(filename))


def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEOS


def allow_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError as e:
        raise Exception("Sorry, deleted failed due to OSError." + str(e))


def listdir(path):
    if secure_filename(path):
        return os.listdir(path)


def uploadFile(file):
    filename_source = ''
    if file.filename == '':
        return Exception("sorry, No file part")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if file and allowed_video(file.filename):
            filename_source = os.path.join(VIDEOS_TRAINS_PATH, filename)
        elif file and allow_image(file.filename):
            filename_source = os.path.join(IMAGES_TRAINS_PATH, filename)
        try:
            file.save(filename_source)
        except Exception as e:
            print('写文件失败' + str(e))
            raise Exception(str(e))


CHUNK_SIZE = 8192


def read_file_chunks(path):
    with open(path, 'rb') as fd:
        while 1:
            buf = fd.read(CHUNK_SIZE)
            if buf:
                yield buf
            else:
                break


def video_stream(filename):
    file_path = os.path.join(VIDEOS_LABELS_PATH, filename)
    if not os.path.exists(file_path):
        return Response("File not found", 404)

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    if range_header:
        bytes_range = range_header.replace("bytes=", "").split("-")
        start = int(bytes_range[0]) if bytes_range[0] else 0
        end = int(bytes_range[1] if bytes_range[1] != '' else 4096) if len(bytes_range) > 1 else file_size - 1

        if start >= file_size or end >= file_size:
            return Response("Requested Range Not Satisfiable", 416)

        content_length = end - start + 1
        response = Response(
            open(file_path, 'rb').read()[start:end + 1],
            206,
            mimetype='video/mp4',
            headers={
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': content_length,
                'Content-Disposition': f'inline; filename="{filename}"'
            }
        )
    else:
        response = send_from_directory(VIDEOS_LABELS_PATH, filename, mimetype='video/mp4')

    return response


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER=UPLOAD_FOLDER_PATH,
        PROJECT_ROOT_PATH=PROJECT_ROOT_PATH,
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
        # response.headers['Accept-Ranges'] = 'bytes'
        return response

    @app.get('/reset')
    def reset():
        try:
            delete_files_in_directory(IMAGES_LABELS_PATH)
            delete_files_in_directory(IMAGES_TRAINS_PATH)
            return jsonify({'status': '200', 'message': 'Reset Successful!'})
        except OSError as e:
            return e

    @app.get('/labels')
    def show_directory_labels():
        try:
            filenames = listdir(IMAGES_LABELS_PATH)
            return filenames
        except IOError as e:
            return e

    @app.get('/trains')
    def show_directory_trains():
        try:
            filenames = listdir(IMAGES_TRAINS_PATH)
            return filenames
        except IOError as e:
            return e

    @app.get('/detection/<path:filename>')
    def show_file_detection(filename):
        if secure_filename(filename):
            return send_from_directory(os.path.abspath(IMAGES_LABELS_PATH), filename, as_attachment=True)

    @app.get('/training/<path:filename>')
    def show_file_training(filename):
        if secure_filename(filename):
            return send_from_directory(os.path.abspath(IMAGES_TRAINS_PATH), filename, as_attachment=True)

    @app.post('/detection/labelsAbsolutePath')
    def get_labels_absolute_path():
        data = request.json
        filenames = data['labels']
        if not filenames:
            return jsonify({'error': 'No labels found'})

        directory = [send_from_directory(os.path.abspath(IMAGES_LABELS_PATH), filename, as_attachment=True) for filename
                     in filenames if filename != '']

        return directory

    @app.get('/detection')
    def detection():
        filenames = os.listdir(IMAGES_TRAINS_PATH)
        filename_sources = [os.path.join(IMAGES_TRAINS_PATH, filename) for filename in filenames if filename != '']
        if len(filename_sources) <= 0:
            return jsonify({'status': 400, 'message': 'No labels found'})

        results = detector.detect_images(filename_sources)
        detect_results = []
        cate = {0.0: '龟裂', 1.0: '杂质', 2.0: '斑块', 3.0: '点蚀', 4.0: '轧制氧化层', 5.0: '划痕'}
        for index, result in enumerate(results):
            result.save(filename=os.path.join(IMAGES_LABELS_PATH, 'label_' + filenames[index]))
            boxes = result.numpy().boxes
            cls = boxes.cls
            conf = boxes.conf
            data = boxes.data
            xywh = boxes.xywh
            category = [cate.get(cls_item) for cls_item in cls]
            # cls = np.array(cls)
            # conf = np.array(conf)
            # data = np.array(data)
            # masks = result.masks
            # keypoints = result.keypoints
            # probs = result.probs
            # obb = result.obb
            # detect_result = {'boxes': boxes, 'masks': masks, 'keypoints': keypoints, 'probs': probs, 'obb': obb}
            # detect_result =
            detect_result = {'cls': cls, 'conf': conf, 'data': data, 'xywh': xywh, 'category': category}
            detect_results.append(detect_result)
        return jsonify({'status': 200, 'imageInfo': '检测成功', 'detectResults': pd.Series(detect_results).to_json()})

    @app.get('/videoDetection/<path:filename>')
    def video_detection(filename):
        if secure_filename(filename):
            filename_sources = os.path.join(VIDEOS_TRAINS_PATH, filename)

            name = (filename.split('.')[0] if ('.' in filename and len(filename.split('.')) >= 1) else filename) + '.webm'
            filename_target = os.path.join(VIDEOS_LABELS_PATH, name)

            detector.detect_video(video_path=filename_sources, save_path=filename_target)
            # for index, result in enumerate(results):
            #     result.save(filename=os.path.join(VIDEOS_LABELS_PATH, 'label_' + filename))
            return jsonify({'status': 200, 'imageInfo': '检测成功', 'videoUrl': name})
        return jsonify({'status': 400, 'imageInfo': '检测失败'})

    @app.route('/videos/<path:name>')
    def serve_video(name):
        filename = os.path.join(VIDEOS_LABELS_PATH, name)
        return send_file(filename, as_attachment=False)

        # filename_sources = os.path.join(VIDEOS_LABELS_PATH, filename)
        # fp = Path(filename_sources)
        # if fp.exists():
        #     return Response(
        #         stream_with_context(read_file_chunks(fp)),
        #         headers={
        #             'Content-Disposition': f'attachment; filename={filename}'
        #         },
        #         content_type='video/mp4'
        #     )
        # else:
        #     raise Exception('File not found')

    @app.post('/uploadFile')
    def upload_file():
        # filename_source = ''
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     if file and allowed_video(file.filename):
        #         filename_source = os.path.join(VIDEOS_TRAINS_PATH, filename)
        #     elif file and allow_image(file.filename):
        #         filename_source = os.path.join(IMAGES_TRAINS_PATH, filename)
        #     file.save(filename_source)
        try:
            uploadFile(file)
        except Exception as e:
            return str(e)
        return jsonify({'status': 200, 'imageInfo': '上传文件成功'})

    @app.post('/uploadFiles')
    def upload_files():
        if 'files[]' not in request.files:
            flash('No files part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            try:
                uploadFile(file)
            except Exception as e:
                return str(e)
        return jsonify({'status': 200, 'imageInfo': '上传文件成功'})

    return app


if __name__ == '__main__':
    app = create_app()
    with  app.app_context():
        detector = Detector()
    app.run(host=host, port=port)
