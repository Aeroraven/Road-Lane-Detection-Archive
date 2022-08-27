import albumentations as albu
from albumentations import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def to_tensor_2(x, **kwargs):
    return x.transpose(1, 0).astype('float32')


def get_preprocessing(preprocessing_fn):
    if preprocessing_fn is None:
        return albu.Lambda(image=to_tensor, mask=to_tensor)
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation():
    train_transform = [
    ]
    return albu.Compose(train_transform)


def video_scale(dest_size=256):
    _transform = [
        albu.Resize(dest_size, dest_size)
    ]
    return albu.Compose(_transform)


def video_scale_ex(x=256, y=256):
    _transform = [
        albu.Resize(x, y)
    ]
    return albu.Compose(_transform)


def image_minmax_scale(image):
    max_v = np.max(image)
    min_v = np.min(image)
    return (image - min_v) / (max_v - min_v)


def preset_processing(params):
    if params == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise Exception("Unsupported weights")
    transform_train = albu.Compose([
        albu.Normalize(mean=mean, std=std),
        albu.Lambda(image=to_tensor, mask=to_tensor_2),
    ])
    return transform_train


def preset_processing_2(params):
    if params == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise Exception("Unsupported weights")
    transform_train = albu.Compose([
        albu.Normalize(mean=mean, std=std),
    ])
    return transform_train
