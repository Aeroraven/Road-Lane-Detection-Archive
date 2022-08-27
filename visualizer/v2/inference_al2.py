# Author of this file: HugePotatoMonster
# https://github.com/HugePotatoMonster/
import matplotlib.pyplot as plt
import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.special
import time
import os

from backend.socket_utils import base64_to_image
from visualizer.yolo.yolo_vis import YOLOV5_ONNX


def preprocess(img):
    img_transforms = transforms.Compose([
        transforms.Resize((480, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    processed_img = img_transforms(img)
    return processed_img


def visualise(img, result, griding_num=200, img_w=800, img_h=480, transparent=False):
    print("Image Shape",img.shape)
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    cls_num_per_lane = 18
    row_anchor_w = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    row_anchor = [int(1.0 * row_anchor_w[i] / 288.0 * 480) for i in range(len(row_anchor_w))]
    print(row_anchor)
    out_j = result.data.cpu().numpy()
    out_j = out_j[0]

    out_j_2 = np.argmax(out_j, axis=0)
    out_j_3 = np.transpose(out_j, (1, 2, 0))
    print("out_j: ", out_j.shape)
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 480)) - 1)
                    if transparent:
                        cv2.circle(img, ppp, 5, (0, 255, 0,255), -1)
                    else:
                        cv2.circle(img, ppp, 5, (0, 255, 0), -1)

    return img


class UL2InferenceSession:
    session = None
    yolo_session:YOLOV5_ONNX = None

    def start_session(self, path, yolo_path):
        UL2InferenceSession.session = onnxruntime.InferenceSession(path,
                                                                   providers=['TensorrtExecutionProvider',
                                                                              'CUDAExecutionProvider',
                                                                              'CPUExecutionProvider'])
        UL2InferenceSession.yolo_session = YOLOV5_ONNX(yolo_path)

    def inference(self, imgs, rawimg = False):
        start_time = time.time()
        if not rawimg:
            img = cv2.imread(imgs)
        else:
            img = imgs
        img = cv2.resize(img, (800, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, channel = img.shape
        input = Image.fromarray(img)
        input = preprocess(input).numpy()
        input = input.reshape(1, 3, 480, 800)
        res_yolo = UL2InferenceSession.yolo_session.infer(imgs,rawimg)
        result_o = UL2InferenceSession.session.run([], {'input': input})
        seg_pred = result_o[1][0]
        result = result_o[0]
        result = np.transpose(result, (0, 1, 3, 2))
        print("TYPE:",type(result))
        result = torch.from_numpy(np.array(result))
        res_yolo = res_yolo.squeeze()
        #res_yolo = np.transpose(res_yolo,(1,2,0))
        print("R",res_yolo.shape)
        res_yolo = cv2.resize(res_yolo,(800, 480))
        res_yolo = cv2.cvtColor(res_yolo,cv2.COLOR_BGR2RGB)
        vis = visualise(res_yolo, result).reshape(img_h, img_w, 3)
        seg_pred = (np.argmax(seg_pred, axis=0) == 1)
        vis[:, :, 0][seg_pred] = 255
        vis[:, :, 1][seg_pred] = 0
        vis[:, :, 2][seg_pred] = 0
        return vis

    def stream_inference(self, source, user_id):
        while True:
            data = source.get_info(user_id)
            if data != "":
                try:
                    img = cv2.imread(base64_to_image(data, user_id))
                except:
                    img = None
            else:
                img = None
            if type(img) is np.ndarray:
                img = cv2.resize(img, (800, 480))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_h, img_w, channel = img.shape
                input = Image.fromarray(img)
                input = preprocess(input).numpy()
                input = input.reshape(1, 3, 480, 800)
                res_yolo = UL2InferenceSession.yolo_session.infer(img, True, True)
                result_o = UL2InferenceSession.session.run([], {'input': input})
                seg_pred = result_o[1][0]
                result = result_o[0]
                result = np.transpose(result, (0, 1, 3, 2))
                print("TYPE:", type(result))
                result = torch.from_numpy(np.array(result))
                res_yolo = res_yolo.squeeze()
                # res_yolo = np.transpose(res_yolo,(1,2,0))
                print("R", res_yolo.shape)
                res_yolo = cv2.resize(res_yolo, (800, 480))
                # res_yolo = cv2.cvtColor(res_yolo, cv2.COLOR_BGR2RGB)
                vis = visualise(res_yolo, result,transparent=True).reshape(img_h, img_w, 4)
                seg_pred = (np.argmax(seg_pred, axis=0) == 1)
                vis[:, :, 0][seg_pred] = 255
                vis[:, :, 1][seg_pred] = 0
                vis[:, :, 2][seg_pred] = 0
                vis[:, :, 3][seg_pred] = 255
                #vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGRA)
            else:
                vis = cv2.imread(r"C:\Users\huang\Pictures\TEST.jpg").astype("uint8")
            vis = cv2.imencode('.png', vis)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + vis + b'\r\n')


if __name__ == "__main__":
    fw = UL2InferenceSession()
    fw.start_session(r"../../static/model/alwenmk2b.onnx")
    im = fw.inference(r"C:\arrow_image\image\_Record001_Camera 5_170927_063836972_Camera_5.jpg")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    cv2.imshow("W",im)
    cv2.waitKey(0)