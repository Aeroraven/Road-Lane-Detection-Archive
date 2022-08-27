import cv2
import numpy as np
import pandas as pd

from utils.vision import *
from utils.vision_mangled import pUSiVycu as homography_estimator


def get_swift_config(**kwargs):
    w = kwargs['swift_w']
    h = kwargs['swift_h']
    c = kwargs['swift_c']
    wm = kwargs['swift_w_modifier']
    iw = kwargs['image_scale_w']
    ih = kwargs['image_scale_h']
    return w, h, c, wm, iw, ih


def swift_coord2image(input_coord, **kwargs):
    # Input Coord: (C*H)
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    mt = np.zeros((input_coord.shape[1], sw)).astype("float")
    for i in range(input_coord.shape[0]):
        for j in range(input_coord.shape[1]):
            if input_coord[i, j] < 0:
                continue
            mt[j, int(input_coord[i, j] / swm)] = 1.0
    return mt


def swift_mcoord2image(input_mcoord, **kwargs):
    # Input Coord: (C*H*(W+1))
    color_list = [
        np.array([1, 0.1, 0.1]),
        np.array([1, 1, 0.1]),
        np.array([0.1, 0.1, 1]),
        np.array([0.1, 1, 0.1]),
    ]
    # Preset
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    mt = np.zeros((sh, sw, 3)).astype("float")
    input_coord = np.argmax(input_mcoord, axis=2)
    for i in range(sc):
        for j in range(sh):
            if input_coord[i, j] == sw:
                continue
            mt[j, int(input_coord[i, j]), :] = color_list[i]
    return mt.astype("float")


def swift_homography(input_mimage, org_shape, **kwargs):
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    src_points = [
        np.array([0, 0]),
        np.array([0, sh - 1]),
        np.array([sw - 1, 0]),
        np.array([sw - 1, sh - 1]),
    ]
    dst_points = [
        np.array([0, ih * imod]),
        np.array([0, ih - 1]),
        np.array([(sw - 1) * swm * iw / ow, ih * imod]),
        np.array([(sw - 1) * swm * iw / ow, (ih - 1)]),
    ]
    homograph_mat = homography_estimator(src_points, dst_points)
    input_mimage_byte = (input_mimage * 255).astype("uint8")
    warpped_out = cv2.warpPerspective(input_mimage_byte, homograph_mat, (iw, ih), flags=cv2.INTER_NEAREST)
    return warpped_out


def swift_homograph_transform(x, y, h):
    homogeneous_coord = np.ones((3, 1))
    homogeneous_coord[0, 0] = x
    homogeneous_coord[1, 0] = y
    homogeneous_coord[2, 0] = 1
    transformed_coord = np.matmul(h, homogeneous_coord)
    return transformed_coord[0, 0] / transformed_coord[2, 0], transformed_coord[1, 0] / transformed_coord[2, 0]


def swift_poly_calc(x, coef):
    ret = 0
    xs = 1
    for i in range(len(coef)):
        ret += xs * coef[len(coef) - i - 1]
        xs *= x
    return ret


def swift_curve_fitting(input_coord, org_image, org_shape, curve_deg=1, width=3,
                        accuracy=5, **kwargs):
    # Input Coord: (C*H)
    # Preset
    color_list = [
        np.array([255, 0, 0]).astype("uint8"),
        np.array([255, 255, 0]).astype("uint8"),
        np.array([0, 0, 255]).astype("uint8"),
        np.array([0, 255, 0]).astype("uint8"),
        np.array([0, 255, 255]).astype("uint8"),
    ]
    # Start
    if len(input_coord.shape) == 3:
        d_flag = True
        input_coord = np.argmax(input_coord, axis=-1)
    else:
        d_flag = False
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    curves = [[] for _ in range(sc)]
    poly = []
    ransac = []
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    src_points = [
        np.array([0, 0]),
        np.array([0, sh - 1]),
        np.array([sw - 1, 0]),
        np.array([sw - 1, sh - 1]),
    ]
    dst_points = [
        np.array([0, ih * imod]),
        np.array([0, ih - 1]),
        np.array([(sw - 1) * swm * iw / ow, ih * imod]),
        np.array([(sw - 1) * swm * iw / ow, (ih - 1)]),
    ]
    homograph_mat = homography_estimator(src_points, dst_points)
    drw_image = org_image.copy()
    fy = 1e100
    for i in range(input_coord.shape[0]):
        for j in range(input_coord.shape[1]):
            if input_coord[i, j] < 0:
                continue
            if d_flag:
                if input_coord[i, j] == sw:
                    continue
                curves[i].append(list(swift_homograph_transform(int(input_coord[i, j]), j, homograph_mat)))
            else:
                curves[i].append(list(swift_homograph_transform(int(input_coord[i, j] / swm), j, homograph_mat)))
        mx = np.array(curves[i])
        if mx.shape[0] == 0 or mx.shape[0] <= 5:
            poly.append([])
            ransac.append([])
            continue
        tx = mx[:, 0]
        ty = mx[:, 1]
        for k in range(len(ty)):
            fy = min(fy, ty[k])
        # poly.append(np.polyfit(ty, tx, curve_deg))
        ransac.append(ransac_polyfit(ty, tx, curve_deg))
        for j in range(int(fy) * accuracy + accuracy, ih * accuracy, 1):
            # xc = swift_poly_calc(j / accuracy, poly[i])
            xc = ransac_poly_predict(ransac[i], j / accuracy)
            if int(xc) >= iw or int(xc) < 0:
                pass
            else:
                for k in range(-width, width):
                    if iw > int(xc) + k >= 0:
                        drw_image[int(np.rint(j / accuracy)) - 1, int(xc) + k, :] = color_list[i]
    return drw_image


def alwen_output_displaying(input_mcoord, pr_mask, org_image, org_shape, curve_deg=1, width=3, accuracy=5, **kwargs):
    org_image = swift_point_displaying(input_mcoord, org_image, org_shape, curve_deg, width, accuracy, **kwargs)
    pr_mask_copy = pr_mask[pr_mask > 0.5] = 1
    org_image[0][pr_mask_copy == 1] = 0
    org_image[1][pr_mask_copy == 1] = 255
    org_image[2][pr_mask_copy == 1] = 0
    return org_image


def swift_point_displaying(input_mcoord, org_image, org_shape, hgs, hgt, hga, curve_deg=1, width=3,
                           accuracy=5, **kwargs):
    color_list = [
        np.array([255, 0, 0]).astype("uint8"),
        np.array([255, 255, 0]).astype("uint8"),
        np.array([0, 0, 255]).astype("uint8"),
        np.array([0, 255, 0]).astype("uint8"),
        np.array([0, 255, 255]).astype("uint8"),
    ]
    # Preset
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    mt = np.zeros((sh, sw, 3)).astype("float")
    input_coord = np.argmax(input_mcoord, axis=2)
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    src_points = [
        np.array([0, 0]),
        np.array([0, sh - 1]),
        np.array([sw - 1, 0]),
        np.array([sw - 1, sh - 1]),
    ]
    dst_points = [
        np.array([0, ih * imod]),
        np.array([0, ih - 1]),
        np.array([(sw - 1) * swm * iw / ow, ih * imod]),
        np.array([(sw - 1) * swm * iw / ow, (ih - 1)]),
    ]
    homograph_mat = homography_estimator(src_points, dst_points)
    drw_image = org_image.copy()
    for i in range(sc):
        cnt = 0
        pt_list_x = []
        pt_list_y = []
        for j in range(sh):

            if input_coord[i, j] == sw:
                continue
            pt_list_x.append(j)
            pt_list_y.append(input_coord[i, j])
            cnt += 1
        if cnt <= 7:
            continue
        p_coef = abs(pd.Series(pt_list_y).corr(pd.Series(pt_list_x), method="pearson"))
        if p_coef < 0.3:
            print("Drop Lane", i, p_coef)
            continue

        for j in range(sh):
            if input_coord[i, j] == sw:
                continue
            tx, ty = swift_homograph_transform(int(input_coord[i, j]), j, homograph_mat)
            for dx in range(-width, width):
                for dy in range(-width, width):
                    if 0 <= int(ty + dy) < ih and 0 <= (tx + dx) < iw:
                        drw_image[int(ty + dy), int(tx + dx), :] = color_list[i]
    return drw_image


def swiftregr_curve_fitting(input_coord, org_image, org_shape, curve_deg=1, width=3,
                            accuracy=5, **kwargs):
    # Input Coord: (3C+4+2C)
    # Preset
    color_list = [
        np.array([255, 0, 0]).astype("uint8"),
        np.array([255, 255, 0]).astype("uint8"),
        np.array([0, 0, 255]).astype("uint8"),
        np.array([0, 255, 0]).astype("uint8"),
        np.array([0, 255, 255]).astype("uint8"),
    ]
    # Start
    if len(input_coord.shape) == 3:
        d_flag = True
        input_coord = np.argmax(input_coord, axis=-1)
    else:
        d_flag = False
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    curves = [[] for _ in range(sc)]
    poly = []
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    src_points = [
        np.array([0, 0]),
        np.array([0, sh - 1]),
        np.array([sw - 1, 0]),
        np.array([sw - 1, sh - 1]),
    ]
    dst_points = [
        np.array([0, ih * imod]),
        np.array([0, ih - 1]),
        np.array([(sw - 1) * swm * iw / ow, ih * imod]),
        np.array([(sw - 1) * swm * iw / ow, (ih - 1)]),
    ]
    homograph_mat = homography_estimator(src_points, dst_points)
    drw_image = org_image.copy()
    A = input_coord[3 * sc]
    B = input_coord[3 * sc + 1]
    C = input_coord[3 * sc + 2]
    P = input_coord[3 * sc + 3]
    D = input_coord[0:sc]
    AL = input_coord[sc:2 * sc]
    BE = input_coord[sc * 2:sc * 3]
    PR = input_coord[sc * 3 + 4:]
    PR = np.reshape(PR, (sc, 2))
    for i in range(sc):
        if PR[i, 0] > PR[i, 1]:
            # continue
            pass
        for j in range(sh):
            print("ALBE", AL[i], BE[i])
            if j < AL[i] or j > BE[i]:
                pass
            pr_x = A * ((j - P) ** 2) + B / (j - P) + C + D[i] * (j - P)
            curves[i].append(list(swift_homograph_transform(pr_x, j, homograph_mat)))
        mx = np.array(curves[i])
        if mx.shape[0] == 0 or mx.shape[0] < 2:
            poly.append([])
            continue
        tx = mx[:, 0]
        ty = mx[:, 1]

        for j in range(len(tx)):
            xc = tx[j]
            yc = ty[j]
            if int(xc) >= iw or int(xc) < 0:
                pass
            else:
                for k in range(-accuracy, accuracy):
                    for l in range(-accuracy, accuracy):
                        if int(xc + l) >= iw or int(xc + l) < 0:
                            continue
                        if int(yc + k) >= ih or int(yc + k) < 0:
                            continue
                        drw_image[int(yc + k) - 1, int(xc + l) - 1, :] = color_list[i]

    return drw_image


def polyregr_curve_fitting(input_coord, org_image, org_shape, est_deg=4, curve_deg=1, width=3,
                           accuracy=5, **kwargs):
    # Input Coord: (x_pr,x_re,x_st,x_vh)
    # Preset
    color_list = [
        np.array([255, 0, 0]).astype("uint8"),
        np.array([255, 255, 0]).astype("uint8"),
        np.array([0, 0, 255]).astype("uint8"),
        np.array([0, 255, 0]).astype("uint8"),
        np.array([0, 255, 255]).astype("uint8"),
    ]
    # Start
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    curves = [[] for _ in range(sc)]
    poly = []
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    drw_image = org_image.copy()
    PR = input_coord[:2 * sc]
    for i in range(sc):
        CO = np.reshape(input_coord[2 * sc + est_deg * i:2 * sc + est_deg * (i + 1)], (1, est_deg))
        for j in range(ih):
            iy = np.reshape(np.array([j ** k for k in range(est_deg)]), (est_deg, 1))
            pr_x = np.matmul(CO, iy)
            curves[i].append(list([pr_x, j]))
            print(pr_x, j)
        mx = np.array(curves[i])
        if mx.shape[0] == 0 or mx.shape[0] < 2:
            poly.append([])
            continue
        tx = mx[:, 0]
        ty = mx[:, 1]

        for j in range(len(tx)):
            xc = tx[j]
            yc = ty[j]
            if int(xc) >= iw or int(xc) < 0:
                pass
            else:
                for k in range(-accuracy, accuracy):
                    for l in range(-accuracy, accuracy):
                        if int(xc + l) >= iw or int(xc + l) < 0:
                            continue
                        if int(yc + k) >= ih or int(yc + k) < 0:
                            continue
                        drw_image[int(yc + k) - 1, int(xc + l) - 1, :] = color_list[i]

    return drw_image


def swift_point_displaying_v2(input_mcoord, org_image, org_shape, hgs=10.0, hgt=59.0, hga=59.0, curve_deg=1, width=3,
                              accuracy=5, **kwargs):
    color_list = [
        np.array([255, 0, 0]).astype("uint8"),
        np.array([255, 255, 0]).astype("uint8"),
        np.array([0, 0, 255]).astype("uint8"),
        np.array([0, 255, 0]).astype("uint8"),
        np.array([0, 255, 255]).astype("uint8"),
    ]
    # Preset
    sw, sh, sc, swm, iw, ih = get_swift_config(**kwargs)
    print("SC:",sc)
    mt = np.zeros((sh, sw, 3)).astype("float")
    input_coord = np.argmax(input_mcoord, axis=2)
    imod = kwargs['tu_height_modifier']
    oh, ow, _ = org_shape
    src_points = [
        np.array([0, 0]),
        np.array([0, sh - 1]),
        np.array([sw - 1, 0]),
        np.array([sw - 1, sh - 1]),
    ]
    dst_points = [
        np.array([0, ih * (hgs/hga)]),
        np.array([0, ih * (hgt/hga)]),
        np.array([iw, ih * (hgs/hga)]),
        np.array([iw, ih * (hgt/hga)]),
    ]
    homograph_mat = homography_estimator(src_points, dst_points)
    drw_image = org_image.copy()
    for i in range(sc):
        cnt = 0
        pt_list_x = []
        pt_list_y = []
        for j in range(sh):

            if input_coord[i, j] == sw:
                continue
            pt_list_x.append(j)
            pt_list_y.append(input_coord[i, j])
            cnt += 1
        if cnt <= 5:
            print("Drop Lane", i, cnt, "By Num")
            continue
        p_coef = abs(pd.Series(pt_list_y).corr(pd.Series(pt_list_x), method="pearson"))
        if p_coef < 0.3:
            print("Drop Lane", i, p_coef)
            continue

        for j in range(sh):
            if input_coord[i, j] == sw:
                continue
            tx, ty = swift_homograph_transform(int(input_coord[i, j]+0.5), j, homograph_mat)
            tx += swm*0.5
            for dx in range(-width, width):
                for dy in range(-width, width):
                    if 0 <= int(ty + dy) < ih and 0 <= (tx + dx) < iw:
                        drw_image[int(ty + dy), int(tx + dx), :] = color_list[i]
    return drw_image
