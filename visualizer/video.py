import cv2 as cv
import jpeg4py

from runner.inference import ProductionModel
from utils.preprocessing import *
from utils.utility import *


class VideoVisualizer:
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
        self.preprocess = get_preprocessing(smp.encoders.get_preprocessing_fn(self.fargs["encoder_arch"],
                                                                              self.fargs["pretrained_encoder_weight"]))

    def video_play(self):
        tp = 0
        if self.video_path.endswith(".avi"):
            cap = cv.VideoCapture(self.video_path)
        else:
            cap = os.listdir(self.video_path)
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
                org_frame = jpeg4py.JPEG(self.video_path + "/" + cap[counter - 1]).decode()
                org_frame = cv2.cvtColor(org_frame, cv2.COLOR_RGBA2BGR)
            frame = video_scale_ex(self.fargs["image_scale_h"],
                                   self.fargs["image_scale_w"])(image=org_frame)['image']
            disp_frame = video_scale_ex(self.fargs["output_shape_x"],
                                        self.fargs["output_shape_y"])(image=org_frame)["image"]
            '''
            input_frame = self.preprocess(image=frame)['image']
            pred_lane = self.model.infer(input_frame)

            lane_mask = pred_lane[1]
            lane_mask[lane_mask > 0.5] = 1
            lane_mask = video_scale_ex(self.fargs["output_shape_x"],
                                       self.fargs["output_shape_y"])(image=lane_mask)["image"]
            disp_frame = np.transpose(disp_frame, (2, 0, 1))

            disp_frame[0][lane_mask == 1] = 0
            disp_frame[1][lane_mask == 1] = 255
            disp_frame[2][lane_mask == 1] = 0

            disp_frame = np.transpose(disp_frame, (1, 2, 0))

            disp_frame = cv2.putText(disp_frame, "FPS: " + str(int((counter / (time.time() - last_time)) * 100) / 100),
                                     (0, 20),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
            disp_frame = cv2.putText(disp_frame, "Frames: " + str(counter), (0, 40),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
             '''
            # last_time = time.time()
            cv.imshow('input', disp_frame)
            if cv.waitKey(1) == ord('q'):
                break
        if tp == 0:
            cap.release()
        cv.destroyAllWindows()

    def video_generate(self):
        video = cv2.VideoWriter("./ckpt/production/demo.avi", 0, 30, (self.fargs["output_shape_x"],
                                                                      self.fargs["output_shape_y"]))
        cap = cv.VideoCapture(self.video_path)
        counter = 0
        last_time = time.time()
        frames_num = cap.get(7)
        with tqdm(total=frames_num, desc="Exporting video", file=sys.stdout, ascii=True) as t:
            while cap.isOpened():
                counter = counter + 1
                ret, org_frame = cap.read()
                if not ret:
                    break
                frame = video_scale_ex(self.fargs["image_scale_h"],
                                       self.fargs["image_scale_w"])(image=org_frame)['image']
                disp_frame = video_scale_ex(self.fargs["output_shape_x"],
                                            self.fargs["output_shape_y"])(image=org_frame)["image"]
                input_frame = self.preprocess(image=frame)['image']
                pred_lane = self.model.infer(input_frame)

                lane_mask = pred_lane[1]
                lane_mask[lane_mask > 0.5] = 1
                lane_mask = video_scale_ex(self.fargs["output_shape_x"],
                                           self.fargs["output_shape_y"])(image=lane_mask)["image"]
                disp_frame = np.transpose(disp_frame, (2, 0, 1))

                disp_frame[0][lane_mask == 1] = 0
                disp_frame[1][lane_mask == 1] = 255
                disp_frame[2][lane_mask == 1] = 0

                disp_frame = np.transpose(disp_frame, (1, 2, 0))
                disp_frame = cv2.putText(disp_frame, "FPS: " + str(int((1 / (time.time() - last_time)) * 100) / 100),
                                         (0, 20),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
                disp_frame = cv2.putText(disp_frame, "Frames: " + str(counter), (0, 40),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), thickness=1)
                last_time = time.time()
                video.write(disp_frame)
                t.update(1)
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    visualizer = VideoVisualizer(r'C:\Users\huang\Desktop\ff\05081544_0305', **get_config_json())
    visualizer.video_play()
