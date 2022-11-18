import glob
import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from skimage import io, img_as_ubyte
from tqdm import tqdm

from utils import *
from xai.adasise import AdaSISE
from xai.density_map import DensityMap
from xai.drise import DRISE
from xai.gradcam import GradCAM, GradCAMPlusPlus
from xai.kde import KDE
from xai.lime_method import LIME
from xai.rise import RISE

warnings.filterwarnings('ignore')
start = datetime.now()


def main(args):
    # ---------------------------------Parameters-------------------------------------
    img_rs, output_tensor, last_conv_tensor, grads, num_sample, NMS = None, None, None, None, None, None
    config_xAI = get_config(args.config_path)
    config_models = get_config(config_xAI['Model']['file_config'])
    image_dict = {}
    sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
            config_models[0]['model_path'])
    threshold = config_xAI['Model']['threshold']

    # create array to save results for 5 metrics    
    drop_rate = []
    inc = []
    ebpg_ = []
    bbox_ = []
    iou_ = []

    # Run xAI for each image
    for j in tqdm(sorted(glob.glob(f'{args.image_path}/*.jpg'))):
        # Load image from input folder and extract ground-truth labels from xml file
        image = cv2.imread(j)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.reshape(1, img.shape[0], img.shape[1], 3)
        name_img = os.path.basename(j).split('.')[0]
        mu = np.mean(image)
        h_img, w_img = img.shape[:2]
        y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run( \
        [detection_boxes, detection_scores, num_detections, detection_classes], \
        feed_dict={img_input: image})
        if y_p_num_detections == 0:
            continue
        # load the saliency map
        cam_map = np.load(os.path.join(args.output_numpy, f"{args.method}_{name_img}.npy"))
        if args.method == 'eLRP':
            cam_map = cam_map[:, :, 2] - cam_map[:, :, 0]
            cam_map = abs(cam_map)
            cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min()) 
        elif args.method == 'D-RISE':
            map = 0
            for cam in cam_map:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
                map += cam
            cam_map = map
        # Coordinates of the predicted boxes
        box_predicted = []
        mask = np.zeros_like(cam_map)
        for i in range(int(y_p_num_detections[0])):
            x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
            y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
            box_predicted.append([x1,y1,x2,y2])
            mask[y1:y2, x1:x2] = 1  
        # ---------------------------DROP-INC----------------------------
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())
        invert = ((img / 255) * np.dstack([cam_map]*3)) * 255
        bias = mu * (1 -cam_map)
        masked = (invert + bias[:, :, np.newaxis]).astype(np.uint8)
        masked = masked[None, :]
        p_boxes, p_scores, p_num_detections, p_classes = sess.run( \
        [detection_boxes, detection_scores, num_detections, detection_classes], \
        feed_dict={img_input: masked})
        prob = y_p_scores[0][:int(y_p_num_detections[0])].sum()
        prob_ex = p_scores[0][:int(p_num_detections[0])].sum()
        if prob < prob_ex:
            inc.append(1)
        drop = max((prob - prob_ex) / prob, 0)
        drop_rate.append(drop)
        # ---------------------------Localization evaluation----------------------------
        bbox_.append(bounding_boxes(box_predicted, cam_map))
        ebpg_.append(energy_point_game(box_predicted, cam_map))
        iou_.append(IoU(mask, cam_map))
    # print results with eps = 1e-10 avoid the case where the denominator is 0
    print("Drop rate:", sum(drop_rate)/(len(drop_rate)+1e-10))
    print("Increases", sum(inc)/(len(inc)+1e-10))
    print("EBPG:", sum(ebpg_)/(len(ebpg_)+1e-10))
    print("Bbox:", sum(bbox_)/(len(bbox_)+1e-10))
    print("IoU:", sum(iou_)/(len(iou_)+1e-10))

if __name__ == '__main__':
    arguments = get_parser()
    main(arguments)
    print(f'Total training time: {datetime.now() - start}')    
    