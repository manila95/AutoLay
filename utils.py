import math

import numpy as np



def lane_scores(pred, gt, iou_thresh=0.5, pix_thresh=400, size=256):
    ego_pred = np.zeros((size, size))
    ego_pred[pred==1] = 1
    ego_gt = np.zeros((size, size))
    ego_gt[gt==1] = 1
    ego_IU = mean_IU(ego_pred, ego_gt)
    ego_AP = mean_precision(ego_pred, ego_gt)

    detections, count = 0., 0.
    true_lane_ids = list(set(gt.ravel()))
    pred_lane_ids = list(set(pred.ravel()))
    for true_id in true_lane_ids:
        if true_id < 1:
            continue
        for pred_id in pred_lane_ids:
            if pred_id < 1:
                continue
            lane_pred = np.zeros((size, size))
            lane_gt = np.zeros((size, size))
            lane_pred[pred==pred_id] = 1
            lane_gt[gt==true_id] = 1
            lane_iou = mean_IU(lane_pred, lane_gt)
            if lane_iou[1] > iou_thresh:
                detections += 1
                break

        #if sum(lane_pred.ravel()) < pix_thresh:
        #    continue

        #count += 1
        #lane_iou = mean_IU(lane_pred, lane_gt)
        #if lane_iou[1] > iou_thresh:
        #    detections += 1


    if len(true_lane_ids) == 1:
        print("True lane ids: ", true_lane_ids)
        true_lane_ids += [1]
    if len(pred_lane_ids) == 1:
        print("Pred lane ids: ", pred_lane_ids)
        pred_lane_ids += [1]
    AP = detections / float(len(pred_lane_ids) - 1)
    Recall = detections / float(len(true_lane_ids)-1)

    return ego_IU, ego_AP, AP, Recall, detections, float(len(pred_lane_ids)-1), float(len(true_lane_ids)-1)





def lane_scores_old(pred, gt, iou_thresh=0.5, pix_thresh=400, size=256):
    ego_pred = np.zeros((size, size))
    ego_pred[pred==1] = 1
    ego_gt = np.zeros((size, size))
    ego_gt[gt==1] = 1
    ego_IU = mean_IU(ego_pred, ego_gt)
    ego_AP = mean_precision(ego_pred, ego_gt)

    detections = 0.
    true_lane_ids = list(set(gt.ravel()))
    pred_lane_ids = list(set(pred.ravel()))
    for lane_id in true_lane_ids:
        if lane_id < 1:
            continue

        lane_pred = np.zeros((size, size))
        lane_gt = np.zeros((size, size))
        lane_pred[pred==lane_id] = 1
        lane_gt[gt==lane_id] = 1

        #if sum(lane_pred.ravel()) < pix_thresh:
        #    continue

        lane_iou = mean_IU(lane_pred, lane_gt)
        if lane_iou[1] > iou_thresh:
            detections += 1 

    
    #if len(true_lane_ids) == 1:
    #    print("True lane ids: ", true_lane_ids)
    #    true_lane_ids += [1]
    #if len(pred_lane_ids) == 1:
    #    print("Pred lane ids: ", pred_lane_ids)
    #    pred_lane_ids += [1]
    AP = detections / float(len(pred_lane_ids)-1)
    Recall = detections / float(len(true_lane_ids)-1)

    return ego_IU, ego_AP, detections, float(len(pred_lane_ids)-1), float(len(true_lane_ids)-1)








def mean_precision(eval_segm, gt_segm):

    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    mAP = [0] * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        n_ij = np.sum(curr_eval_mask)
        val = n_ii / float(n_ij)
        if math.isnan(val):
            mAP[i] = 0.
        else:
            mAP[i] = val
    # print(mAP)
    return mAP


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    return IU


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
