# Author of this file: HugePotatoMonster
# https://github.com/HugePotatoMonster/

import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.special
import time
import os

from matplotlib import pyplot as plt

from backend.socket_utils import base64_to_image


def preprocess(img):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    processed_img = img_transforms(img)
    return processed_img


def visualise(img, result, griding_num=200, img_w=1640, img_h=590):
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    cls_num_per_lane = 18
    row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

    out_j = result[0].data.cpu().numpy()
    out_j = out_j[0]
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
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(img, ppp, 5, (0, 255, 0, 255), -1)

    return img


class ULInferenceSession:
    session = None

    def start_session(self, path):
        ULInferenceSession.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider'])

    def inference(self, img):
        start_time = time.time()
        img = cv2.imread(img)
        img = cv2.resize(img, (1640, 590))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(img)
        input = preprocess(input).numpy()
        input = input.reshape(1, 3, 288, 800)
        result = ULInferenceSession.session.run([], {'input': input})
        result = torch.from_numpy(np.array(result))
        vis = visualise(img, result).reshape(590, 1640, 3)
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
                img = cv2.resize(img, (1640, 590))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input = Image.fromarray(img)
                input = preprocess(input).numpy()
                input = input.reshape(1, 3, 288, 800)
                result = ULInferenceSession.session.run([], {'input': input})
                result = torch.from_numpy(np.array(result))
                img = np.zeros((590,1640,4)).astype("uint8")
                vis = visualise(img, result).reshape(590, 1640, 4).astype("uint8")
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGRA)
            else:
                vis = cv2.imread(r"C:\Users\huang\Pictures\TEST.jpg").astype("uint8")
            vis = cv2.imencode('.png', vis)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + vis + b'\r\n')


if __name__ == "__main__":
    fw = ULInferenceSession()
    fw.start_session(r"../../static/model/ultra.onnx")
    im = fw.inference(r"C:\Users\huang\Desktop\ff\nw\05081544_0305-002045.jpg")
    plt.imshow(im)
    plt.show()
