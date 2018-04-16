import numpy as np
import cv2

def calculate_iou(img_mask, gt_mask):
    """
    This method calculates the Intersection over Union value between image mask
    and ground-truth mask. 
    
    Args:
        img_mask (numpy array): image mask
        gt_mask (numpy array): ground-truth mask
    Returns:
        iou (int): Intersection over Union value
    """
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou

def get_iou(gt_mask, old_mask, new_mask):
    """
    This method calls calculate_iou() on the old image and new image.
    
    Args:
        gt_mask (numpy array): ground-truth mask
        old_mask (numpy array): old image mask 
        new_mask (numpy array): new image mask
    Returns:
        iou, new_iou (int): old and new Intersection over Union value
    """
    iou = calculate_iou(old_mask, gt_mask)
    new_iou = calculate_iou(new_mask, gt_mask)
    return iou, new_iou
