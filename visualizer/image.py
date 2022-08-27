import cv2
import numpy as np
from matplotlib import pyplot as plt

from datasets.seg_v2 import SegmentationDatasetVal
from runner.inference import ProductionModel
from utils.preprocessing import image_minmax_scale
from utils.utility import *
from visualizer.swift_proc import *


class ImageVisualizer:
    @staticmethod
    def seg_image_visualize(**kwargs):
        fargs = {}
        for key in kwargs.keys():
            fargs[key] = kwargs[key]

        print("Preparing runner")
        model = ProductionModel(fargs["model_path"], get_model_type(fargs["model_path"]), fargs['infer_device'])
        print("Model ready")
        preproc = get_preproc_func(**fargs)
        train_dataset = SegmentationDataset(fargs["test_image_path"], fargs["test_mask_path"], [0, 1],
                                            image_size_w=fargs["image_scale_w"],
                                            image_size_h=fargs["image_scale_h"],
                                            numpy_dataset=fargs["numpy_dataset"],
                                            preprocessing=preproc,
                                            image_only=True,
                                            tu_simple_trapezoid_filter=fargs[
                                                                           'dataset_type'] == 'tu_simple_trapezoid_seg')
        idx = random.randint(0, len(train_dataset) - 1)
        print("Chosen Idx", idx)
        if len(train_dataset[idx]) == 2:
            image, mask = train_dataset[idx]
            random.seed(time.time())
            pred = model.infer(image)[1]
            mask = np.transpose(mask, (1, 0))
            prows, pcols = (2, 2)
            plt.subplot(prows, pcols, 1)
            plt.title("Mask-PR")
            plt.imshow(pred)
            plt.subplot(prows, pcols, 2)
            plt.title("Mask-GT")
            print(mask.shape)
            plt.imshow(mask)
            image = np.transpose(np.squeeze(image), (1, 2, 0))
            plt.subplot(prows, pcols, 3)
            plt.title("Input")
            plt.imshow(image_minmax_scale(image))
        else:
            image = train_dataset[idx]
            random.seed(time.time())
            pred = model.infer(image)[1]
            prows, pcols = (2, 2)
            plt.subplot(prows, pcols, 1)
            plt.title("Mask-PR")
            plt.imshow(pred)
            plt.subplot(prows, pcols, 2)
            plt.title("Mask-GT")
            image = np.transpose(np.squeeze(image), (1, 2, 0))
            plt.subplot(prows, pcols, 3)
            plt.title("Input")
            plt.imshow(image_minmax_scale(image))

        # CV Algos
        imcv = np.minimum(np.rint(image_minmax_scale(image) * 255), 255).astype("uint8")
        imcv1 = cv2.Canny(imcv, 50, 300)
        plt.subplot(prows, pcols, 4)
        plt.title("Input-Canny")
        plt.imshow(imcv1)

        plt.show()

    @staticmethod
    def swift_visualizer(**kwargs):
        fargs = kwargs
        fargs['_runtime_testmode'] = True
        print("Preparing runner")
        model = ProductionModel(fargs["model_path"], get_model_type(fargs["model_path"]), fargs['infer_device'])
        print("Model ready")
        preproc = get_preproc_func(**fargs)
        train_dataset = get_train_dataset(preproc, [0, 1], **fargs)
        idx = random.randint(0, len(train_dataset) - 1)
        print("Chosen Idx", idx)
        image, coord, image_org, img_org_shape = train_dataset[idx]
        print(image.shape)
        pred = model.infer(image)
        image = np.transpose(image, (1, 2, 0))
        im_coord = swift_coord2image(coord, **kwargs)
        im_pcoord = swift_mcoord2image(pred, **kwargs)
        im_pwarped = swift_homography(im_pcoord, img_org_shape, **kwargs)
        im_gt_curve_fit = swift_curve_fitting(coord, image_org, img_org_shape, **kwargs)
        im_pr_curve_fit = swift_curve_fitting(pred, image_org, img_org_shape, **kwargs)
        prows, pcols = (2, 3)
        plt.subplot(prows, pcols, 1)
        plt.title("Input")
        plt.imshow(image_org)
        plt.subplot(prows, pcols, 2)
        plt.title("Label: GT")
        plt.imshow(im_coord)
        plt.subplot(prows, pcols, 3)
        plt.title("Label: PR")
        plt.imshow(im_pcoord)

        # Masking
        im_pwarped_mask_3 = (im_pwarped > 0).astype("uint8")
        im_pwarped_mask_3n = 1 - im_pwarped_mask_3
        im_pwarped_3 = im_pwarped
        im_mixed = im_pwarped_3 + im_pwarped_mask_3n * image_org
        plt.subplot(prows, pcols, 4)
        plt.title("Output")
        plt.imshow(im_mixed)

        plt.subplot(prows, pcols, 5)
        plt.title("CurveFit: GT")
        plt.imshow(im_gt_curve_fit)

        plt.subplot(prows, pcols, 6)
        plt.title("CurveFit: PR")
        plt.imshow(im_pr_curve_fit)

        plt.show()

    @staticmethod
    def alwenmk2_visualizer(**kwargs):
        fargs = kwargs
        fargs['_runtime_testmode'] = True
        print("Preparing runner")
        model_container = get_model(mode="test",**fargs)
        model = ProductionModel(fargs["model_path"], get_model_type(fargs["model_path"]), fargs['infer_device'],
                                compatible_mode=False, output_tasks=2, model_container=model_container)
        print("Model ready")
        preproc = get_preproc_func(**fargs)
        train_dataset = SegmentationDatasetVal(fargs["test_image_path"], fargs["test_mask_path"], [0, 1],
                                               image_size_w=fargs["image_scale_w"],
                                               image_size_h=fargs["image_scale_h"],
                                               numpy_dataset=fargs["numpy_dataset"],
                                               preprocessing=preproc,
                                               image_only=True,
                                               tu_simple_trapezoid_filter=fargs[
                                                                              'dataset_type'] == 'tu_simple_trapezoid_seg')
        idx = random.randint(0, len(train_dataset) - 1)
        print("Chosen Idx", idx)
        org_img, image = train_dataset[idx]
        pred, seg_pred = model.infer(image)
        seg_pred = (np.argmax(seg_pred,axis=0) == 1)
        print("Shape", pred.shape)
        pred = np.transpose(pred, (1, 2, 0))
        image = np.transpose(image, (1, 2, 0))
        im_pcoord = swift_point_displaying_v2(pred, org_img, org_img.shape, **kwargs)
        prows, pcols = (1, 2)
        plt.subplot(prows, pcols, 1)
        plt.title("Input")
        plt.imshow(org_img)
        plt.subplot(prows, pcols, 2)
        plt.title("Label: PR")
        # im_pcoord[:, :, 0][seg_pred] = 255
        # im_pcoord[:, :, 1][seg_pred] = 0
        # im_pcoord[:, :, 2][seg_pred] = 0
        plt.imshow(im_pcoord)

        plt.show()
