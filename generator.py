import json
import math
import os

import cfg
from utils import *


def process_data(label_file, use_aug=False):
    label = json.load(open(label_file, 'r'))

    if "imageData" in label:
        image = string_to_image(label['imageData'], format='rgb')
    else:
        dir = os.path.dirname(label_file)
        image_path = os.path.join(dir, label['imagePath'])
        image = cv2.imread(image_path)

    classes = []
    list_points = []
    boxes = []

    for obj in label['shapes']:
        class_name = obj['label']
        points = np.array(obj['points'])
        box = points_to_bbox(points)

        classes.append(class_name)
        list_points.append(points)
        boxes.append(box)

    points = np.array(list_points, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)

    image, boxes, points = image_preprocessing(image.copy(), (cfg.input_image_w, cfg.input_image_h), gt_boxes=boxes,
                                               gt_keypoints=points)

    # prepare store tensor
    hm = np.zeros((cfg.output_h, cfg.output_w, cfg.n_classes), dtype=np.float32)
    wh = np.zeros((cfg.max_objs, 2), dtype=np.float32)
    reg = np.zeros((cfg.max_objs, 2), dtype=np.float32)
    ind = np.zeros(cfg.max_objs, dtype=np.int32)
    reg_mask = np.zeros(cfg.max_objs, dtype=np.int32)

    hm_kp = np.zeros((cfg.output_h, cfg.output_w, cfg.n_kps), dtype=np.float32)
    kps = np.zeros((cfg.max_objs, cfg.n_kps, 2), dtype=np.float32)
    kps_mask = np.zeros((cfg.max_objs * cfg.n_kps), dtype=np.int32)# TODO: remove
    kps_ind = np.zeros((cfg.max_objs, cfg.n_kps), dtype=np.int32)# TODO: remove
    kp_offset = np.zeros((cfg.max_objs * cfg.n_kps, 2), dtype=np.float32)
    kp_ind = np.zeros((cfg.max_objs * cfg.n_kps), dtype=np.int32)
    kp_mask = np.zeros((cfg.max_objs * cfg.n_kps), dtype=np.int32)

    # convert to feature scale
    boxes = boxes / cfg.down_ratio
    points = points / cfg.down_ratio

    for idx in range(boxes.shape[0]):
        class_id = cfg.class2id[classes[idx]]
        bbox = boxes[idx]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[:, :, class_id], ct_int, radius)
        wh[idx] = 1. * w, 1. * h
        ind[idx] = ct_int[1] * cfg.output_w + ct_int[0]
        reg[idx] = ct - ct_int
        reg_mask[idx] = 1

        for j in range(cfg.n_kps):
            kps[idx, j] = points[idx, j, :2] - ct_int # TODO: FROM HERE
            # kps_mask[idx, j * 2:j * 2 + 2] = 1 # TODO: remove
            pt_int = points[idx, j, :2].astype(np.int32)
            kp_offset[idx * cfg.n_kps + j] = points[idx, j, : 2] - pt_int
            kp_ind[idx * cfg.n_kps + j] = pt_int[1] * cfg.output_w + pt_int[0]
            kp_mask[idx * cfg.n_kps + j] = 1

            draw_umich_gaussian(hm_kp[:, :, j], pt_int, radius)

    # if cfg.debug:
    #     print(reg_mask)
    #     print(ind)
    #     print(kp_ind)
    #     print(kps_mask)
    #     cv2.imshow("image", image)
    #     cv2.imshow("hm", hm)
    #     cv2.imshow("hm_kp", hm_kp[:, :, 1:])
    #     cv2.waitKey(0)
    return image, hm, wh, reg, reg_mask, ind, hm_kp, kps, kps_ind, kps_mask, kp_offset, kp_ind, kp_mask # TODO remove kps_ind, kps_mask


def get_data(batch_files):
    batch_image = np.zeros((cfg.batch_size, cfg.input_image_h, cfg.input_image_w, 3), dtype=np.float32)

    batch_hm = np.zeros((cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_classes), dtype=np.float32)
    batch_wh = np.zeros((cfg.batch_size, cfg.max_objs, 2), dtype=np.float32)
    batch_reg = np.zeros((cfg.batch_size, cfg.max_objs, 2), dtype=np.float32)
    batch_reg_mask = np.zeros((cfg.batch_size, cfg.max_objs), dtype=np.int32)
    batch_ind = np.zeros((cfg.batch_size, cfg.max_objs), dtype=np.int32)

    batch_hm_kp = np.zeros((cfg.batch_size, cfg.output_h, cfg.output_w, cfg.n_kps), dtype=np.float32)
    batch_kps = np.zeros((cfg.batch_size, cfg.max_objs, cfg.n_kps, 2), dtype=np.float32)
    batch_kps_ind = np.zeros((cfg.batch_size, cfg.max_objs, cfg.n_kps), dtype=np.int32)# TODO: remove
    batch_kps_mask = np.zeros((cfg.batch_size, cfg.max_objs, cfg.n_kps), dtype=np.int32)# TODO: remove
    batch_kp_offset = np.zeros((cfg.batch_size, cfg.max_objs * cfg.n_kps, 2), dtype=np.float32)
    batch_kp_ind = np.zeros((cfg.batch_size, cfg.max_objs * cfg.n_kps), dtype=np.int32)
    batch_kp_mask = np.zeros((cfg.batch_size, cfg.max_objs * cfg.n_kps), dtype=np.int32)

    for num, line in enumerate(batch_files):
        image, hm, wh, reg, reg_mask, ind, hm_kp, kps, kps_ind, kps_mask, kp_offset, kp_ind, kp_mask = process_data( # TODO: remove kps_ind and kps_mask
            line,
            use_aug=cfg.use_aug)
        batch_image[num, :, :, :] = image

        batch_hm[num, :, :, :] = hm
        batch_wh[num, :, :] = wh
        batch_reg[num, :, :] = reg
        batch_reg_mask[num, :] = reg_mask
        batch_ind[num, :] = ind

        batch_hm_kp[num, :, :, :] = hm_kp
        batch_kps[num, :, :] = kps
        # batch_kps_mask[num, :] = kps_mask # TODO: remove
        batch_kp_offset[num, :] = kp_offset
        batch_kp_ind[num, :] = kp_ind
        batch_kp_mask[num, :] = kp_mask

    return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_hm_kp, batch_kps, batch_kps_mask, batch_kp_offset, batch_kp_ind, batch_kp_mask # TODO: remove batch_kps_mask


if __name__ == '__main__':
    get_data([
        "/media/data_it/Data_set/images/tima/fe_crawled_pdf/verified/label_corner/11FP6B926d9VmFxuMiZ4Ya7Mh08J5OoqR_0.json"],
        False)
