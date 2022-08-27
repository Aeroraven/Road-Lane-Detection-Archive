from utils.utility import get_config_json
from visualizer.image import ImageVisualizer
from visualizer.video import VideoVisualizer
from visualizer.video_swift import VideoVisualizerSwift

if __name__ == "__main__":
    exec_type = 0  # 0 - Image Visualizer / 1 - Video Visualizer
    if exec_type == 0:
        kwargs = get_config_json()
        if kwargs['model_arch'] == 'scnn' or kwargs['model_arch'] == 'unet' or kwargs['model_arch'] == 'yolop' \
            or kwargs['model_arch'] == 'arrowfcn':
            ImageVisualizer.seg_image_visualize(**kwargs)
        elif kwargs['model_arch'] == "swift" :
            ImageVisualizer.swift_visualizer(**kwargs)
        elif kwargs['model_arch'] == "alwen2":
            ImageVisualizer.alwenmk2_visualizer(**kwargs)
        else:
            raise Exception("Unknown model arch")
    elif exec_type == 1:
        kwargs = get_config_json()
        if kwargs['model_arch'] == 'scnn' or kwargs['model_arch'] == 'unet' or \
                kwargs['model_arch'] == 'yolop':
            visualizer = VideoVisualizer(
                r'E:\Nf\val',
                **get_config_json())
            visualizer.video_play()
        else:
            visualizer = VideoVisualizerSwift(
                r'E:\MNIST\train_set\clips\0313-1',
                **get_config_json())
            visualizer.video_play(**kwargs)
