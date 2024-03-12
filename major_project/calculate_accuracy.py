import numpy as np
import cv2


def get_corners(x, y, w, l, yaw):
    # Calculate corner points of the bounding box
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    half_w = w / 2
    half_l = l / 2
    corners = np.array([
        [-half_l * cos_yaw - half_w * sin_yaw + x, -half_l * sin_yaw + half_w * cos_yaw + y],
        [-half_l * cos_yaw + half_w * sin_yaw + x, -half_l * sin_yaw - half_w * cos_yaw + y],
        [half_l * cos_yaw + half_w * sin_yaw + x, half_l * sin_yaw - half_w * cos_yaw + y],
        [half_l * cos_yaw - half_w * sin_yaw + x, half_l * sin_yaw + half_w * cos_yaw + y]
    ])
    return corners


def calculate_iou(box1, box2):
    # Calculate intersection points
    box1 = box1.astype(np.int32)
    box2 = box2.astype(np.int32)
    intersection_pts = cv2.convexHull(np.concatenate((box1, box2), axis=0))

    # Calculate area of intersection and union
    intersection_area = cv2.contourArea(intersection_pts)
    area_box1 = cv2.contourArea(box1)
    area_box2 = cv2.contourArea(box2)
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0

    return abs(iou)
