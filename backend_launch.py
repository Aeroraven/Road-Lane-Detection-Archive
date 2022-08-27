import json
import os

import cv2
from flask import session, Response
from flask import Flask, request, make_response
from flask import render_template
from flask_cors import cross_origin

from backend.image_infer import serv_image_infer_alwenmk2, ServModelCache, video_infer
from backend.result_factory import ResultFactory
from visualizer.v2.inference import ULInferenceSession

import sortedcontainers as sc

from visualizer.v2.inference_al2 import UL2InferenceSession

app = Flask(__name__)
app.secret_key = b'\x1d\xa7mZ\xb6q\xb3\xe89|\x94?y\xd4\x95$'


class CacheItem:
    def __init__(self, data, order):
        self.data = data
        self.order = order

    @staticmethod
    def compare(x):
        return x.order


class SessCache:
    def __init__(self):
        self.rm: dict[int, sc.SortedList] = {}
        self.uid = 0

    def allocate_user(self):
        self.rm[self.uid] = sc.SortedList(key=CacheItem.compare)
        self.uid += 1
        return self.uid - 1

    def upload_info(self, sid, data, order):
        # print("Added,", sid, order, len(self.rm[sid]))
        self.rm[sid].add(CacheItem(data, order))

    def get_remain(self, sid):
        return len(self.rm[sid])

    def get_info(self, sid):
        # print("Remain,", sid, len(self.rm[sid]))
        while len(self.rm[sid]) <= 1:
            for i in range(1000):
                pass
            continue
        res = self.rm[sid][0].data
        if len(self.rm[sid]) > 1:
            self.rm[sid].pop(0)
        return res


sc_dict = SessCache()


@app.route("/")
@app.route("/index.html")
@cross_origin()
def hello_world():
    return "HelloWorld"


@app.route("/handShake")
@cross_origin()
def hello_world_2():
    return "1950641"


@app.route("/api")
@cross_origin()
def hello_world_2b():
    return "HelloWorld-2"


@app.route("/api/core/imageInfer", methods=['post'])
@cross_origin()
def hello_world_3():
    basedir = os.path.abspath(os.path.dirname(__file__))
    img = request.files.get('image')
    path = basedir + "/static/image/"
    file_path = path + img.filename
    img.save(file_path)
    output_path = serv_image_infer_alwenmk2(file_path, basedir)
    return ResultFactory.build_success_result(output_path), 200, {'Content-Type': 'application/json',
                                                                  "Access-Control-Allow-Headers": 'Content-Type',
                                                                  'Access-Control-Allow-Credentials': 'true',
                                                                  'Access-Control-Allow-Methods': "POST"
                                                                  }


@app.route('/api/videoUpload', methods=['post'])
@cross_origin()
def video_feed():
    basedir = os.path.abspath(os.path.dirname(__file__))
    img = request.files.get('image')
    path = basedir + "/static/video/"
    file_path = path + img.filename
    img.save(file_path)
    return ResultFactory.build_success_result(file_path), 200, {'Content-Type': 'application/json',
                                                                "Access-Control-Allow-Headers": 'Content-Type',
                                                                'Access-Control-Allow-Credentials': 'true',
                                                                'Access-Control-Allow-Methods': "POST"
                                                                }


@app.route('/api/videoInfer', methods=['get'])
@cross_origin()
def video_play():
    basedir = os.path.abspath(os.path.dirname(__file__))
    file_path = request.args.get('path')
    return Response(video_infer(file_path, basedir), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/image/<file>")
@cross_origin()
def get_assets(file: str = None):
    path = os.path.abspath(os.path.dirname(__file__))
    with open(path + "/static/image/" + file, "rb") as f:
        ret = f.read()
    tp = "none"
    file = file.lower()
    if file.endswith(".png"):
        tp = 'image/png'
    if file.endswith(".jpg"):
        tp = 'image/jpg'
    if file.endswith(".jfif"):
        tp = 'image/jfif'
    response = make_response(ret)
    response.headers['Content-Type'] = tp
    return response


@app.route('/api/camUpload', methods=['post'])
@cross_origin()
def cam_uploading():
    user = int(request.args.get('id'))
    time = int(request.args.get('time'))
    param = request.form.get('image')
    sc_dict.upload_info(user, param, time)
    return ResultFactory.build_success_result("succ"), 200, {'Content-Type': 'application/json',
                                                             "Access-Control-Allow-Headers": 'Content-Type',
                                                             'Access-Control-Allow-Credentials': 'true',
                                                             'Access-Control-Allow-Methods': "POST"
                                                             }


@app.route('/api/camShow', methods=['get'])
@cross_origin()
def cam_show():
    user_id = int(request.args.get('id'))
    return Response(UL2InferenceSession().stream_inference(sc_dict, user_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camJoin', methods=['post'])
@cross_origin()
def cam_join():
    uid = sc_dict.allocate_user()
    return ResultFactory.build_success_result(uid), 200, {'Content-Type': 'application/json',
                                                          "Access-Control-Allow-Headers": 'Content-Type',
                                                          'Access-Control-Allow-Credentials': 'true',
                                                          'Access-Control-Allow-Methods': "POST"
                                                          }


@app.route('/api/camRemain', methods=['post'])
@cross_origin()
def cam_remain():
    user_id = int(request.args.get('id'))
    return ResultFactory.build_success_result(sc_dict.get_remain(user_id)), 200, {'Content-Type': 'application/json',
                                                                                  "Access-Control-Allow-Headers": 'Content-Type',
                                                                                  'Access-Control-Allow-Credentials': 'true',
                                                                                  'Access-Control-Allow-Methods': "POST"
                                                                                  }


ServModelCache.start_deploy(os.path.abspath(os.path.dirname(__file__)))
ULInferenceSession().start_session(os.path.abspath(os.path.dirname(__file__)) + r"\static\model\ultra.onnx")
UL2InferenceSession().start_session(os.path.abspath(os.path.dirname(__file__)) + r"\static\model\best.onnx",
                                    os.path.abspath(os.path.dirname(__file__)) + r"\static\model\yolov5s.onnx",)

if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)
