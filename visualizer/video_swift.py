import re

import cv2 as cv
import jpeg4py

from runner.inference import ProductionModel
from utils.preprocessing import *
from utils.utility import *
from visualizer.swift_proc import *
from utils.augmentation import *

class VideoVisualizerSwift:
    def __init__(self, video_path: str, **kwargs):
        self.model_path = kwargs["model_path"]
        self.video_path = video_path
        self.fargs = {}
        for key in kwargs.keys():
            self.fargs[key] = kwargs[key]
        print("Preparing runner")
        if self.model_path == "":
            self.model = None
        else:
            self.model = ProductionModel(self.fargs['model_path'], get_model_type(self.fargs['model_path']),
                                         self.fargs['visualizer_device'])
        print("Model ready")
        self.preprocess = get_preproc_func(**kwargs)
        self.model_arch = kwargs['model_arch']

    def video_play(self, **kwargs):
        print("Play starts")
        tp = 0
        if self.video_path.endswith(".avi") or self.video_path.endswith(".mp4"):
            cap = cv.VideoCapture(self.video_path)
        else:
            cap = []
            capv = os.walk(self.video_path)
            for i,j,k in capv:
                k = sorted(k, key=lambda x: int(re.search(r"^(\d+)", x).group(1)), reverse=False)
                for r in k:
                    cap.append(i+"\\"+r)
            tp = 1
        counter = 0
        last_time = time.time()
        while (tp == 0 and cap.isOpened()) or (tp == 1 and counter != len(cap)):
            counter = counter + 1
            if tp == 0:
                ret, org_frame = cap.read()
                if not ret:
                    break
            else:
                if cap[counter - 1].endswith("txt"):
                    continue
                org_frame = jpeg4py.JPEG(cap[counter - 1]).decode()
                # org_frame = cv2.cvtColor(org_frame, cv2.COLOR_RGBA2BGR)
            org_frame = video_scale_ex(480, 800)(image=org_frame)['image']
            frame = video_scale_ex(self.fargs["image_scale_h"],
                                   self.fargs["image_scale_w"])(image=org_frame)['image']
            input_frame = self.preprocess(image=frame)['image']
            print(input_frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            pred_lane = self.model.infer(input_frame)
            print(pred_lane.shape)
            disp_frame = frame

            if self.model_arch == "swiftregr":
                disp_frame = swiftregr_curve_fitting(pred_lane, frame, org_frame.shape, **kwargs)
            elif self.model_arch == "polylane":
                disp_frame = polyregr_curve_fitting(pred_lane, frame, org_frame.shape, **kwargs)
            else:
                pred_lane = np.transpose(pred_lane,(2,0,1))
                print(pred_lane.shape)
                disp_frame = swift_point_displaying(pred_lane, frame, org_frame.shape, **kwargs)

            disp_frame = cv2.putText(disp_frame, "FPS: " + str(int((counter / (time.time() - last_time)) * 100) / 100),
                                     (0, 20),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
            disp_frame = cv2.putText(disp_frame, "Frames: " + str(counter), (0, 40),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
            # last_time = time.time()
            cv.imshow('input', disp_frame)
            if cv.waitKey(1) == ord('q'):
                break
        if tp == 0:
            cap.release()
        cv.destroyAllWindows()
