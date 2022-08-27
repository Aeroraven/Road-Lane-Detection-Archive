import json
import math
import os
import sys
import time

import segmentation_models_pytorch as smp
import torch.optim
import yaml
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

from datasets.alwen_mix import AlwenMixedDataset
from datasets.apollo import ApolloDataset
from datasets.sampler.mock_sampler import *
from datasets.segmentation import SegmentationDataset
from datasets.tusimple import TuSimpleDataset
from datasets.tusimple_combined import TuSimpleMixedDataset
from datasets.tusimple_v2 import TuSimpleDatasetV2
from datasets.yolop_bdd import BddDataset
from model.alwen import AlwenNet
from model.alwen_mk2 import AlwenNetMK2, AlwenNetMK2Alpha
from model.alwen_mk2b import AlwenNetMK2Beta
from model.alwen_mk2c import AlwenNetMK2Gamma
from model.arrow_fcn import ArrowFCN, ArrowFCNMK2
from model.loss.arrowfcn_loss import ArrowFCNAccMetric, ArrowFCNForeDiceMetric, ArrowFCNIoUMetric, ArrowFCNCELoss
from model.loss.ultra_loss import UltraLaneAccuracy, SoftmaxFocalLoss
from model.polylane_regressor import *
from model.scheduler.sched import MultiStepLR, EmptyLR
from model.spatial_cnn import SpatialCNN
from model.swiftlane_cnn import SwiftLaneCNNV2
from model.swiftlane_cnn_regression import SwiftLaneRegressionCNN
from model.ultra_lane import UltraFastLane
from model.yolop import YOLOPNet
from model.loss.alwen_mk2_loss import *
from utils.alarm import show_warning
from utils.loss import *
from utils.loss import FocalLoss, DiceLoss
from utils.preprocessing import get_preprocessing, preset_processing


def get_workspace_dir(base_path, exp_name):
    base_path_w = base_path
    if base_path_w[-1] == '\\':
        base_path_w = base_path_w[:-1]
    time_stamp = time.strftime("(%Y-%m-%d_%H-%M-%S)", time.localtime())
    dir_name = exp_name + "_" + time_stamp
    full_dir_name = base_path_w + "/" + dir_name
    os.mkdir(full_dir_name)
    return full_dir_name


def save_train_config(work_dir, conf):
    with open(work_dir + "/config.json", "w") as f:
        f.write(json.dumps(conf))


def save_train_log(work_dir, log, name="train_result"):
    with open(work_dir + "/" + name + ".json", "w") as f:
        f.write(json.dumps(log))


def set_seed(seed=3407):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def preload_dataset(dataloader, desc):
    with tqdm(total=len(dataloader), desc=desc, file=sys.stdout, ascii=True) as t:
        for _ in dataloader:
            t.update(1)


def get_config_json(path="./global_strategy.yaml"):
    while os.path.exists(path) == False:
        choice = input("Choose configuration strategy (server/client):")
        if choice == 'server':
            with open(path, "w") as g:
                g.write("server")
                break
        else:
            with open(path, "w") as g:
                g.write("client")
                break
    with open(path, "r") as f:
        data = f.read()
    if data.startswith("server"):
        print("Using config: server")
        return get_config_json_impl('./global_config_server_side.yaml')
    else:
        print("Using config: client")
        return get_config_json_impl('./global_config_client_side.yaml')
    return get_config_json(path)


def get_config_json_impl(path):
    with open(path, "r") as f:
        data = f.read()
    data = yaml.load(data, Loader=yaml.FullLoader)
    return data


def get_model_type(path: str):
    if path.endswith(".pth"):
        return "pytorch"
    elif path.endswith(".onnx"):
        return "onnx"
    else:
        raise Exception("Model type unsupported.")


def get_proc_necessity(arch):
    if arch == "unet":
        return True
    if arch == "scnn":
        return False
    if arch == "swift":
        return False
    if arch == "swiftreg":
        return False
    if arch == "polylane":
        return False
    if arch == "yolop":
        return False
    if arch == "alwen":
        return False
    if arch == "ultra":
        return False
    if arch == "arrowfcn":
        return False
    if arch == "alwen2":
        return False
    raise Exception("Unsupported arch")


def get_loss_func(**kwargs):
    fargs = kwargs
    if fargs["loss"] == "dice":
        loss = DiceLoss()
    elif fargs["loss"] == "focal":
        loss = FocalLoss()
    elif fargs["loss"] == "weighted_ce":
        if fargs["classes"] == 2:
            loss = CrossEntropyLoss(weight=torch.tensor([0.1, 1]).to(fargs["device"]))
        elif fargs["classes"] == 5:
            loss = CrossEntropyLoss(weight=torch.tensor([0.4, 1., 1., 1., 1.]).to(fargs["device"]))
        else:
            raise Exception("Unsupported classes")
    elif fargs["loss"] == "swift_ce":
        loss = SwiftLaneCrossEntropyLoss(c=fargs['swift_c'],
                                         w=fargs['swift_w'],
                                         h=fargs['swift_h'],
                                         w_mod=fargs['swift_w_modifier'])
    elif fargs['loss'] == 'swift_ec':
        loss = SwiftLaneRegressionEcuLoss(c=fargs['swift_c'],
                                          w=fargs['swift_w'],
                                          h=fargs['swift_h'],
                                          w_mod=fargs['swift_w_modifier'])
    elif fargs['loss'] == 'poly_ec':
        loss = PolyLaneExistenceLaneErrorLoss(c=fargs['swift_c'], device=fargs['device'])
    elif fargs['loss'] == 'yolop_loss':
        loss = YOLOPLaneLoss(kwargs['yolop_pad'], kwargs['device'])
    elif fargs['loss'] == 'alwen_loss':
        loss = AlwenMultiTaskLoss(c=fargs['swift_c'],
                                  w=fargs['swift_w'],
                                  h=fargs['swift_h'],
                                  w_mod=fargs['swift_w_modifier'],
                                  device=fargs["device"])
    elif fargs['loss'] == 'ultra_loss':
        loss = SoftmaxFocalLoss(1)
    elif fargs['loss'] == 'arrow_loss':
        loss = ArrowFCNCELoss(fargs['device'])
    elif fargs['loss'] == 'alwen2_loss':
        loss = AL2MultitaskLoss(fargs['device'])
    else:
        raise Exception("Unsupported loss")
    return loss


def get_model(mode="train", **kwargs):
    fargs = kwargs
    if fargs["model_arch"] == "unet":
        print("UseModel", "UNet")
        model = smp.unet.Unet(encoder_name=fargs["encoder_arch"], encoder_weights=fargs["pretrained_encoder_weight"],
                              classes=fargs["classes"], activation=fargs["final_activation"])
    elif fargs["model_arch"] == "scnn":
        print("UseModel", "SCNN")
        if fargs["image_scale_w"] < 512 and fargs['image_scale_h'] < 512:
            show_warning("Backbone might be too large for the given dataset which contains images with small size."
                         "To get the excellent result, increase `image_scale` and rerun `data_processing.py` "
                         "with the unchanged random seed")
        model = SpatialCNN(fargs["image_scale_w"], fargs["image_scale_h"], fargs["classes"],
                           backbone_arch=fargs['scnn_backbone'], backbone_pretrained=True)
    elif fargs["model_arch"] == "swift":
        print("UseModel", "SwiftLane")
        model = SwiftLaneCNNV2(channels=3,
                               height=fargs["image_scale_h"],
                               width=fargs["image_scale_w"],
                               c=fargs['swift_c'],
                               w=fargs['swift_w'],
                               h=fargs['swift_h'])
    elif fargs['model_arch'] == 'swiftreg':
        print("UseModel", "SwiftLaneReg")
        model = SwiftLaneRegressionCNN(channels=3,
                                       height=fargs["image_scale_h"],
                                       width=fargs["image_scale_w"],
                                       c=fargs['swift_c'],
                                       w=fargs['swift_w'],
                                       h=fargs['swift_h'])
    elif fargs['model_arch'] == 'polylane':
        print("UseModel", "PolyLaneSimple")
        model = PolyLaneRegressor(height=fargs["image_scale_h"],
                                  width=fargs["image_scale_w"],
                                  lanes=fargs['swift_c'])
    elif fargs['model_arch'] == 'yolop':
        print("UseModel", "YOLOP Lane Seg Branch")
        model = YOLOPNet()
    elif fargs['model_arch'] == 'alwen':
        print("UseModel", "Multi-task Detector")
        model = AlwenNet(c=fargs['swift_c'], w=fargs['swift_w'])
    elif fargs['model_arch'] == 'ultra':
        print("UseModel", "Ultra Lane Detector")
        model = UltraFastLane(channels=3,
                              height=fargs["image_scale_h"],
                              width=fargs["image_scale_w"],
                              c=fargs['swift_c'],
                              w=fargs['swift_w'],
                              h=fargs['swift_h'])
    elif fargs['model_arch'] == 'arrowfcn':
        print("UseModel", "ArrowFCN")
        model = ArrowFCNMK2()
    elif fargs['model_arch'] == 'alwen2':

        if mode == "train":
            print("UseModel", "Multi-task Detector MK2 Gamma - Full")
            model = AlwenNetMK2Gamma(ih=fargs["image_scale_h"],
                                     iw=fargs["image_scale_w"],
                                     c=fargs['swift_c'],
                                     w=fargs['swift_w'],
                                     h=fargs['swift_h'],
                                     skip_arrow=fargs['skip_arrow'])
        else:
            print("UseModel", "Multi-task Detector MK2 Gamma - DropLSG")
            model = AlwenNetMK2Gamma(ih=fargs["image_scale_h"],
                                     iw=fargs["image_scale_w"],
                                     c=fargs['swift_c'],
                                     w=fargs['swift_w'],
                                     h=fargs['swift_h'],
                                     drop_lsg=True,
                                     skip_arrow=fargs['skip_arrow'])

    else:
        raise Exception("Model architecture is not supported")
    return model


def get_metrics(**kwargs):
    fargs = kwargs
    metrics = {}
    metrics_mask = {}
    if fargs['model_arch'] == 'unet' or fargs['model_arch'] == 'scnn':
        if fargs['loss'] == "weighted_ce":
            metrics = {"dice_acc": DiceMetrics(expand=True, classes=fargs['classes'], device=fargs['device']),
                       "dice_acc_lane": DiceMetrics(selected_channel=1, single_channel=True,
                                                    expand=True, classes=fargs['classes'], device=fargs['device'])}
        else:
            metrics = {"dice_acc": DiceMetrics(),
                       "dice_acc_lane": DiceMetrics(selected_channel=1, single_channel=True)}
    elif fargs['model_arch'] == 'swift':
        metrics = {'grid_acc': SwiftLaneGridAccuracy(c=fargs['swift_c'],
                                                     w=fargs['swift_w'],
                                                     h=fargs['swift_h'],
                                                     w_mod=fargs['swift_w_modifier'])}
        show_warning("There's currently no official implementations of accuracy evaluation methods")
    elif fargs['model_arch'] == 'swiftreg':
        metrics = {'exist_celoss': SwiftLaneRegressionCE(c=fargs['swift_c'],
                                                         w=fargs['swift_w'],
                                                         h=fargs['swift_h'],
                                                         w_mod=fargs['swift_w_modifier'])}
        show_warning("There's currently no official implementations of accuracy evaluation methods")
    elif fargs['model_arch'] == 'polylane':
        metrics = {'exist_celoss': PolyLaneExistCE(c=fargs['swift_c']),
                   # 'lane_dev': PolyLaneExistenceLaneErrorLoss(c=fargs['swift_c'],w1=0,w2=0)
                   }
        show_warning("There's currently no official implementations of accuracy evaluation methods")
    elif fargs['model_arch'] == "yolop":
        metrics = {'iou': YOLOPIouCachedMetric()}
    elif fargs['model_arch'] == 'alwen':
        metrics = {'arrow_iou': AlwenArrowIoUMetric(),
                   'lane_racc': MixedLaneAccuracyMetric(),
                   'arrow_loss': AlwenArrowLossMetric(),
                   'lane_loss': AlwenLaneLossMetric()}
        metrics_mask = {'arrow_iou': AlwenMultiTaskLoss.do_arrow_update,
                        'arrow_loss': AlwenMultiTaskLoss.do_arrow_update,
                        'lane_racc': AlwenMultiTaskLoss.do_lane_update,
                        'lane_loss': AlwenMultiTaskLoss.do_lane_update}
    elif fargs['model_arch'] == 'ultra':
        metrics = {'racc': UltraLaneAccuracy(device=fargs["device"])}
    elif fargs['model_arch'] == 'arrowfcn':
        metrics = {'iou': ArrowFCNIoUMetric(device=fargs["device"]),
                   'acc': ArrowFCNAccMetric(device=fargs['device']),
                   'fore_dice': ArrowFCNForeDiceMetric(device=fargs['device'])}
    elif fargs['model_arch'] == 'alwen2':
        metrics = {'arrow_iou': AL2IoUMetric(),
                   'lane_acc': AL2UltraLaneAccuracy(),
                   'arr_loss': AL2ArrowLossMetric(),
                   'lsg_loss': AL2LsegLossMetric(),
                   'anc_loss': AL2AnchorLossMetric()}
        metrics_mask = {'arrow_iou': AL2MultitaskLoss.do_arrow_update,
                        'lane_acc': AL2MultitaskLoss.do_lane_update,
                        'arr_loss': AL2MultitaskLoss.do_arrow_update,
                        'lsg_loss': AL2MultitaskLoss.do_lane_update,
                        'anc_loss': AL2MultitaskLoss.do_lane_update}
    else:
        raise Exception("Model arch is not supported")
    return metrics, metrics_mask


def get_preproc_func(**kwargs):
    fargs = kwargs
    if get_proc_necessity(fargs["model_arch"]):
        preproc = smp.encoders.get_preprocessing_fn(fargs["encoder_arch"], fargs["pretrained_encoder_weight"])
        preproc = get_preprocessing(preproc)
    else:
        preproc = preset_processing(fargs["pretrained_encoder_weight"])
    return preproc


def get_yolop_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def get_train_dataset(preproc, classes_x, **kwargs):
    fargs = kwargs
    train_dataset = None
    append_org_copy = False
    if '_runtime_testmode' in fargs:
        append_org_copy = True  #
    if fargs['dataset_type'] == 'culane_seg':
        train_dataset = SegmentationDataset(fargs["train_image_path"], fargs["train_mask_path"], classes_x,
                                            image_size_w=fargs["image_scale_w"], image_size_h=fargs["image_scale_h"],
                                            numpy_dataset=fargs["numpy_dataset"],
                                            preprocessing=preproc, mask_augmentation=fargs['mask_augmentation'],
                                            expanding_onehot=fargs['expand_onehot'])
    elif fargs['dataset_type'] == 'tu_simple_trapezoid_seg':
        train_dataset = SegmentationDataset(fargs["train_image_path"], fargs["train_mask_path"], classes_x,
                                            image_size_w=fargs["image_scale_w"], image_size_h=fargs["image_scale_h"],
                                            numpy_dataset=fargs["numpy_dataset"],
                                            preprocessing=preproc, mask_augmentation=fargs['mask_augmentation'],
                                            expanding_onehot=fargs['expand_onehot'],
                                            tu_simple_trapezoid_filter=True)
    elif fargs['dataset_type'] == 'tu_simple':
        train_dataset = TuSimpleDataset(image_path=kwargs['train_image_path'],
                                        image_size_h=kwargs['image_scale_h'],
                                        image_size_w=kwargs['image_scale_w'],
                                        index_subdirectories=kwargs['tu_simple_train_list'],
                                        split_ratio=kwargs['train_test_ratio'],
                                        is_test=False,
                                        preprocessing=preproc,
                                        return_original_copy=append_org_copy,
                                        infer_model=kwargs['tu_simple_trapezoid_model'],
                                        polyfit_format=(kwargs['model_arch'] == 'polylane'))
    elif fargs['dataset_type'] == 'tu_simple_2':
        train_dataset = TuSimpleDatasetV2(image_path=kwargs['train_image_path'],
                                          image_size_h=kwargs['image_scale_h'],
                                          image_size_w=kwargs['image_scale_w'],
                                          index_subdirectories=kwargs['tu_simple_train_list'],
                                          split_ratio=kwargs['train_test_ratio'],
                                          is_test=False,
                                          preprocessing=preproc,
                                          return_original_copy=append_org_copy,
                                          infer_model=kwargs['tu_simple_trapezoid_model'],
                                          polyfit_format=(kwargs['model_arch'] == 'polylane'),
                                          compensate_c=kwargs['swift_c'],
                                          compensate_h=kwargs['swift_h'],
                                          grid_width=kwargs['swift_w_modifier'],
                                          grids=kwargs['swift_w'])
    elif fargs['dataset_type'] == 'bdd_yolop':
        train_dataset = BddDataset(train_path=kwargs['train_image_path'],
                                   mask_path=kwargs['train_mask_path'],
                                   is_train=True,
                                   inputsize=[480, 800],
                                   transform=get_yolop_transforms()
                                   )
    elif fargs['dataset_type'] == 'mock':
        train_dataset = AlwenMockDataset()
    elif fargs['dataset_type'] == 'mixed':
        train_dataset = TuSimpleMixedDataset(image_path=kwargs['train_image_path'],
                                             image_size_h=kwargs['image_scale_h'],
                                             image_size_w=kwargs['image_scale_w'],
                                             index_subdirectories=kwargs['tu_simple_train_list'],
                                             split_ratio=kwargs['train_test_ratio'],
                                             is_test=False,
                                             preprocessing=preproc,
                                             return_original_copy=append_org_copy,
                                             infer_model=kwargs['tu_simple_trapezoid_model'],
                                             polyfit_format=(kwargs['model_arch'] == 'polylane'),
                                             arrow_mask_path=kwargs['train_mask_path_2'],
                                             arrow_path=kwargs['train_image_path_2'])
    elif fargs['dataset_type'] == 'apollo':
        train_dataset = ApolloDataset(image_size_h=kwargs['image_scale_h'],
                                      image_size_w=kwargs['image_scale_w'],
                                      split_ratio=kwargs['train_test_ratio'],
                                      is_test=False,
                                      preprocessing=preproc,
                                      arrow_mask_path=kwargs['test_mask_path_2'],
                                      arrow_path=kwargs['test_image_path_2'])
    elif fargs['dataset_type'] == 'mixed2':
        train_dataset = AlwenMixedDataset(cu_image_path=kwargs['train_image_path_culane'],
                                          cu_seg_mask_path=kwargs['train_mask_path_culane'],
                                          preprocessing=preproc,
                                          is_test=False,
                                          ap_seg_mask_path=kwargs['train_mask_path_2'],
                                          ap_image_path=kwargs['train_image_path_2'],
                                          disable_apollo=kwargs['disable_apollo'])
    else:
        raise Exception("Dataset is not supported")
    return train_dataset


def get_test_dataset(preproc, classes_x, **kwargs):
    fargs = kwargs
    test_dataset = None
    append_org_copy = False
    discard_corruptions = True
    if '_runtime_testmode' in fargs:
        append_org_copy = True
        discard_corruptions = False
    if fargs['dataset_type'] == 'culane_seg':
        test_dataset = SegmentationDataset(fargs["test_image_path"], fargs["test_mask_path"], classes_x,
                                           image_size_w=fargs["image_scale_w"], image_size_h=fargs["image_scale_h"],
                                           numpy_dataset=fargs["numpy_dataset"],
                                           preprocessing=preproc, mask_augmentation=fargs['mask_augmentation'],
                                           expanding_onehot=fargs['expand_onehot'])
    elif fargs['dataset_type'] == 'tu_simple_trapezoid_seg':
        test_dataset = SegmentationDataset(fargs["test_image_path"], fargs["test_mask_path"], classes_x,
                                           image_size_w=fargs["image_scale_w"], image_size_h=fargs["image_scale_h"],
                                           numpy_dataset=fargs["numpy_dataset"],
                                           preprocessing=preproc, mask_augmentation=fargs['mask_augmentation'],
                                           expanding_onehot=fargs['expand_onehot'],
                                           tu_simple_trapezoid_filter=True)
    elif fargs['dataset_type'] == 'tu_simple':
        test_dataset = TuSimpleDataset(image_path=kwargs['test_image_path'],
                                       image_size_h=kwargs['image_scale_h'],
                                       image_size_w=kwargs['image_scale_w'],
                                       index_subdirectories=kwargs['tu_simple_test_list'],
                                       split_ratio=kwargs['train_test_ratio'],
                                       is_test=True,
                                       preprocessing=preproc,
                                       return_original_copy=append_org_copy,
                                       discard_corruption=discard_corruptions,
                                       infer_model=kwargs['tu_simple_trapezoid_model'],
                                       polyfit_format=(kwargs['model_arch'] == 'polylane'))
    elif fargs['dataset_type'] == 'tu_simple_2':
        test_dataset = TuSimpleDatasetV2(image_path=kwargs['train_image_path'],
                                         image_size_h=kwargs['image_scale_h'],
                                         image_size_w=kwargs['image_scale_w'],
                                         index_subdirectories=kwargs['tu_simple_train_list'],
                                         split_ratio=kwargs['train_test_ratio'],
                                         is_test=True,
                                         preprocessing=preproc,
                                         return_original_copy=append_org_copy,
                                         infer_model=kwargs['tu_simple_trapezoid_model'],
                                         polyfit_format=(kwargs['model_arch'] == 'polylane'),
                                         compensate_c=kwargs['swift_c'],
                                         compensate_h=kwargs['swift_h'],
                                         grid_width=kwargs['swift_w_modifier'],
                                         grids=kwargs['swift_w'])
    elif fargs['dataset_type'] == 'bdd_yolop':
        test_dataset = BddDataset(train_path=kwargs['test_image_path'],
                                  mask_path=kwargs['test_mask_path'],
                                  is_train=False,
                                  inputsize=[480, 800],
                                  transform=get_yolop_transforms()
                                  )
    elif fargs['dataset_type'] == 'mock':
        test_dataset = AlwenMockDataset()
    elif fargs['dataset_type'] == 'mixed':
        test_dataset = TuSimpleMixedDataset(image_path=kwargs['test_image_path'],
                                            image_size_h=kwargs['image_scale_h'],
                                            image_size_w=kwargs['image_scale_w'],
                                            index_subdirectories=kwargs['tu_simple_test_list'],
                                            split_ratio=kwargs['train_test_ratio'],
                                            is_test=True,
                                            preprocessing=preproc,
                                            return_original_copy=append_org_copy,
                                            infer_model=kwargs['tu_simple_trapezoid_model'],
                                            polyfit_format=(kwargs['model_arch'] == 'polylane'),
                                            arrow_mask_path=kwargs['test_mask_path_2'],
                                            arrow_path=kwargs['test_image_path_2'])
    elif fargs['dataset_type'] == 'apollo':
        test_dataset = ApolloDataset(image_size_h=kwargs['image_scale_h'],
                                     image_size_w=kwargs['image_scale_w'],
                                     split_ratio=kwargs['train_test_ratio'],
                                     is_test=True,
                                     preprocessing=preproc,
                                     arrow_mask_path=kwargs['test_mask_path_2'],
                                     arrow_path=kwargs['test_image_path_2'])
    elif fargs['dataset_type'] == 'mixed2':
        test_dataset = AlwenMixedDataset(cu_image_path=kwargs['test_image_path_culane'],
                                         cu_seg_mask_path=kwargs['test_mask_path_culane'],
                                         preprocessing=preproc,
                                         is_test=True,
                                         ap_seg_mask_path=kwargs['test_mask_path_2'],
                                         ap_image_path=kwargs['test_image_path_2'],
                                         disable_apollo=kwargs['disable_apollo'])
    else:
        raise Exception("Dataset is not supported")
    return test_dataset


def get_dataloader(dataset, **kwargs):
    if kwargs['dataset_type'] != "mock" and kwargs['dataset_type'] != "mixed" and kwargs['dataset_type'] != "mixed2":
        return torch_data.DataLoader(dataset, batch_size=kwargs["batch_size"],
                                     shuffle=True, num_workers=kwargs['num_workers'])
    else:
        sp = AlwenMockSampler(dataset)
        bs = AlwenMockBatchSampler(sp, kwargs["batch_size"], False)
        return torch_data.DataLoader(dataset,
                                     num_workers=kwargs['num_workers'],
                                     batch_sampler=bs)


def get_train_epoch_type(**kwargs):
    if kwargs['model_arch'] == "scnn":
        return 'segmentation'
    if kwargs['model_arch'] == "unet":
        return 'segmentation'
    if kwargs['model_arch'] == "swift":
        return 'swift_regr'
    if kwargs['model_arch'] == "swiftreg":
        return 'swift_regr'
    if kwargs['model_arch'] == "polylane":
        return 'swift_regr'
    if kwargs['model_arch'] == "yolop":
        return "swift_regr"
    if kwargs['model_arch'] == "alwen":
        return "swift_regr"
    if kwargs['model_arch'] == "ultra":
        return "swift_regr"
    if kwargs['model_arch'] == "arrowfcn":
        return "swift_regr"
    if kwargs['model_arch'] == "alwen2":
        return "swift_regr"
    raise NotImplementedError()


def get_optimizer(cl, param, lr):
    if cl == "adam":
        return torch.optim.Adam(param, lr=lr, betas=(0.937, 0.999), eps=1e-8)
    elif cl == "adam_yolop":
        return torch.optim.Adam(param, lr=lr, betas=(0.937, 0.999))


def get_lr_scheduler(optimizer, **kwargs):
    if kwargs['model_arch'] == 'polylane':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=385)
    elif kwargs['model_arch'] == 'yolop' or kwargs['model_arch'] == 'alwen' or kwargs['model_arch'] == 'arrowfcn' or \
            kwargs['model_arch'] == 'alwen2':
        lf = lambda x: ((1 + math.cos(x * math.pi / kwargs['epochs'])) / 2) * \
                       (1 - kwargs['yolop_lrf']) + kwargs['yolop_lrf']
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        return EmptyLR()


def get_customized_lr_scheduler(optimizer, data_loader_len, **kwargs):
    if kwargs['model_arch'] == 'ultra':
        sched = MultiStepLR(optimizer, [25, 38], 0.1, data_loader_len, 'linear', 695)
    else:
        sched = None
    return sched
