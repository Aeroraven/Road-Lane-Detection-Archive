import cv2

from runner.inference import ProductionModel
from utils.utility import *

import albumentations as albu

from visualizer.swift_proc import swift_point_displaying_v2
from visualizer.v2.inference_al2 import UL2InferenceSession


class ServModelCache:
    alwenmk2_model = None

    @staticmethod
    def start_deploy(base_path):
        print("Loading Models. This may take few minutes, please wait...")
        model_path = base_path + r"\static\model\alwenmk2.onnx"
        ServModelCache.alwenmk2_model = ProductionModel(model_path, get_model_type(model_path), "cpu",
                                                        compatible_mode=False, output_tasks=2)
        print("Models Loaded")


def serv_image_infer_alwenmk2(path: str, base_path: str):
    kwargs = get_config_json_impl(base_path + r"\static\configs\alwenmk2_config.yaml")
    suffix = path.split(".")[-1]
    preproc = get_preproc_func(model_arch="alwen2", pretrained_encoder_weight="imagenet")
    resize = albu.Resize(480, 800)
    image_org = cv2.imread(path)
    sess = UL2InferenceSession()
    vis = sess.inference(path)
    vis = vis.astype("uint8")
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    save_path = path.replace("." + suffix, "_infer" + "." + suffix)
    cv2.imwrite(save_path, vis)
    return save_path.replace(base_path, "").split('/')[-1]


def video_infer(path: str, base_path: str):
    video_path = r'C:\Users\huang\Desktop\Actv\GTA.mp4'
    kwargs = get_config_json_impl(base_path + r"\static\configs\alwenmk2_config.yaml")
    suffix = path.split(".")[-1]
    preproc = get_preproc_func(model_arch="alwen2", pretrained_encoder_weight="imagenet")
    resize = albu.Resize(480, 800)
    vid = cv2.VideoCapture(path)
    while True:
        return_value, frame = vid.read()
        sess = UL2InferenceSession()
        vis = sess.inference(frame,True)
        vis = vis.astype("uint8")
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        image = cv2.imencode('.jpg', vis)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def serv_image_infer_single_alwenmk2(path: str, base_path: str):
    kwargs = get_config_json_impl(base_path + r"\static\configs\alwenmk2_config.yaml")
    suffix = path.split(".")[-1]
    preproc = get_preproc_func(model_arch="alwen2", pretrained_encoder_weight="imagenet")
    resize = albu.Resize(480, 800)
    image_org = cv2.imread(path)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    image_org = resize(image=image_org)['image']
    image = preproc(image=image_org)['image']
    pred, seg_pred = ServModelCache.alwenmk2_model.infer(image)
    seg_pred = (np.argmax(seg_pred, axis=0) == 1)
    pred = np.transpose(pred, (1, 2, 0))
    image = np.transpose(image, (1, 2, 0))
    im_pcoord = swift_point_displaying_v2(pred, image_org, image_org.shape, **kwargs)
    im_pcoord[:, :, 0][seg_pred] = 255
    im_pcoord[:, :, 1][seg_pred] = 0
    im_pcoord[:, :, 2][seg_pred] = 0
    im_pcoord = im_pcoord.astype("uint8")
    # im_pcoord = cv2.cvtColor(im_pcoord, cv2.COLOR_RGB2BGR)
    return im_pcoord


def serv_image_infer_single_alwenmk2_ultra(path: str, base_path: str):
    kwargs = get_config_json_impl(base_path + r"\static\configs\alwenmk2_config.yaml")
    suffix = path.split(".")[-1]
    preproc = get_preproc_func(model_arch="alwen2", pretrained_encoder_weight="imagenet")
    resize = albu.Resize(480, 800)

    image_org = cv2.imread(path)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
    image_org = resize(image=image_org)['image']
    image = preproc(image=image_org)['image']
    pred, seg_pred = ServModelCache.alwenmk2_model.infer(image)
    seg_pred = (np.argmax(seg_pred, axis=0) == 1)
    pred = np.transpose(pred, (1, 2, 0))
    image = np.transpose(image, (1, 2, 0))
    im_pcoord = swift_point_displaying_v2(pred, image_org, image_org.shape, **kwargs)
    im_pcoord[:, :, 0][seg_pred] = 255
    im_pcoord[:, :, 1][seg_pred] = 0
    im_pcoord[:, :, 2][seg_pred] = 0
    im_pcoord = im_pcoord.astype("uint8")
    # im_pcoord = cv2.cvtColor(im_pcoord, cv2.COLOR_RGB2BGR)


    return im_pcoord
