import argparse
import glob
import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from utils import DeepExplain, get_config, get_model, get_info, save_image, gen_cam, GradientMethod
from xai.gradcam import GradCAM, GradCAMPlusPlus
from xai.lime_method import LIME
from xai.rise import RISE

warnings.filterwarnings('ignore')
start = datetime.now()


def get_parser():
    """
    Parse command line arguments
    :return: parser
    """
    parser = argparse.ArgumentParser(description='xAI for thyroid cancer detection')
    parser.add_argument(
            '--config_file',
            default='model/config/model_config.json',
            metavar='FILE',
            help='path to config file',
    )
    parser.add_argument('--method',
                        help='Choose Backpropagation methods: elrp, GradCAM, GradCAM++, RISE, LIME')
    parser.add_argument('--image-path', help='path to input images')
    parser.add_argument('--stage', default='first_stage',
                        help='Choose a stage to visualize: first_stage or second_stage')
    parser.add_argument('--threshold', default=0.6, type=float, help='Threshold of output values to visualize')
    parser.add_argument('--output', help='A file or directory to save output visualizations.')
    return parser


def main(args):
    # ---------------------------------Parameters-------------------------------------
    global img_rs, output_tensor, last_conv_tensor, grads, num_sample, NMS
    config_xAI = get_config(args.config_file)
    config_models = get_config(config_xAI['Model']['file_config'])
    image_dict = {}
    sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes = get_model(
            config_models[0]['model_path'])
    threshold = config_xAI['Model']['threshold']

    # -------------------------Saliency, Grad*Input, eLRP, IntGrad, DeepLIFT-------------------------
    if args.method in ['elrp']:
        img_rs = sess.graph.get_tensor_by_name(config_xAI['Gradient']['target'] + ':0')
        output_tensor = sess.graph.get_tensor_by_name(config_xAI['Gradient']['output'] + ':0')
        with DeepExplain(session=sess) as de:
            explainer = de.get_explainer(args.method, np.sum(output_tensor[0, :, 1:2]), img_rs)

    # ---------------------------------GradCAM, GradCAM++-------------------------------------
    elif args.method in ['GradCAM', 'GradCAM++']:
        last_conv_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM'][args.stage]['target'] + ':0')
        output_tensor = sess.graph.get_tensor_by_name(config_xAI['CAM'][args.stage]['output'] + ':0')

        if args.stage == 'first_stage':
            grads = tf.gradients(np.sum(output_tensor[0, :, 1:2]), last_conv_tensor)[0]
        else:
            NMS = sess.graph.get_tensor_by_name(config_xAI['CAM'][args.stage]['NMS'] + ':0')

    else:
        num_sample = config_xAI[args.method]['num_sample']

    # Run xAI for each image
    for j in tqdm(sorted(glob.glob(f'{args.image_path}/*.jpg'))):
        image = cv2.imread(j)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.reshape(1, img.shape[0], img.shape[1], 3)
        name_img = os.path.basename(j).split('.')[0]
        gr_truth_boxes = get_info(config_xAI['Model']['folder_xml'] + f'{name_img}.xml')

        # First stage of model: Extract 300 boxes
        if args.stage == 'first_stage':
            if args.method in ['elrp']:
                image_rs = sess.run(img_rs, feed_dict={img_input: image})
                baseline = np.zeros_like(image_rs)
                gradient = GradientMethod(sess, img_rs, output_tensor, explainer, baseline)
                image_dict[args.method] = gradient(image, args.method, img_input)
                save_image(image_dict, os.path.basename(j), args.output, index='full_image')

            elif args.method in ['GradCAM', 'GradCAM++']:
                y_p_boxes, y_p_num_detections = sess.run([detection_boxes, num_detections],
                                                         feed_dict={img_input: image})
                boxs = []
                for i in range(int(y_p_num_detections[0])):
                    h_img, w_img = image.shape[1:3]
                    x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
                    y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
                    boxs.append([x1, y1, x2, y2])
                if args.method == 'GradCAM':
                    gradcam = GradCAM(sess, last_conv_tensor, output_tensor)
                    mask = gradcam(image, grads, img_input, args.stage, y_p_boxes)
                    # Save image and heatmap
                    image_dict[args.method], _ = gen_cam(img, mask, boxs)
                    save_image(image_dict, os.path.basename(j), args.output, index='full_image')
                else:
                    grad_cam_plus_plus = GradCAMPlusPlus(sess, last_conv_tensor, output_tensor)
                    mask_plus_plus = grad_cam_plus_plus(image, grads, img_input, args.stage, y_p_boxes)  # cam mask
                    # Save image and heatmap
                    image_dict[args.method], _ = gen_cam(img, mask_plus_plus, boxs)
                    save_image(image_dict, os.path.basename(j), args.output, index='full_image')
        # Second stage of model: Detect final boxes containing the nodule(s)
        else:
            y_p_boxes, y_p_scores, y_p_num_detections = sess.run([detection_boxes, detection_scores, num_detections],
                                                                 feed_dict={img_input: image})

            if args.method in ['RISE']:
                boxs = []
                grid_size = config_xAI['RISE']['grid_size']
                prob = config_xAI['RISE']['prob']
                index = config_xAI['RISE']['index']
                assert y_p_scores[0][index] > args.threshold

                for i in range(int(y_p_num_detections[0])):
                    h_img, w_img = image.shape[1:3]
                    x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
                    y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
                    boxs.append([x1, y1, x2, y2])

                rise = RISE(image=image, sess=sess, grid_size=grid_size, prob=prob, num_samples=num_sample)
                rs = rise.explain(image, index, img_input, detection_boxes, detection_scores, num_detections,
                                  detection_classes)[0]
                image_dict[args.method], _ = gen_cam(img, rs, boxs)
                save_image(image_dict, os.path.basename(j), args.output, index=f'box{index}')

            elif args.method in ['LIME']:
                index = config_xAI['LIME']['index']
                num_features = config_xAI['LIME']['num_features']
                feature_view = 1
                lime = LIME(sess, image=img, indices=index, num_features=num_features)
                image_dict[args.method] = lime.explain(feature_view, num_samples=num_sample)
                save_image(image_dict, os.path.basename(j), args.output, index=f'box{index}')

            elif args.method in ['GradCAM', 'GradCAM++']:
                index = config_xAI['CAM'][args.stage]['index']
                assert y_p_scores[0][index] > args.threshold
                NMS_tensor = sess.run(NMS, feed_dict={img_input: image})
                indices = NMS_tensor[index]
                grads = tf.gradients(output_tensor[0][indices][1], last_conv_tensor)[0]

                if args.method == 'GradCAM':
                    # Run GradCAM and save results
                    gradcam = GradCAM(sess, last_conv_tensor, output_tensor)
                    mask, x1, y1, x2, y2 = gradcam(image,
                                                   grads,
                                                   args.stage,
                                                   y_p_boxes,
                                                   indices=indices,
                                                   index=index,
                                                   y_p_boxes=y_p_boxes) # cam mask
                    # Save image and heatmap
                    image_dict['predict_box'] = img[y1:y2, x1:x2]  # [H, W, C]
                    image_dict[args.method], _ = gen_cam(img[y1:y2, x1:x2], mask)
                    save_image(image_dict, os.path.basename(j), args.output, index=f'2th_stage_box{index}')
                else:
                    # Run GradCAM++ and save results
                    grad_cam_plus_plus = GradCAMPlusPlus(sess, last_conv_tensor, output_tensor)
                    mask_plus_plus, x1, y1, x2, y2 = grad_cam_plus_plus(image,
                                                                        grads,
                                                                        args.stage,
                                                                        y_p_boxes,
                                                                        indices=indices,
                                                                        index=index,
                                                                        y_p_boxes=y_p_boxes)  # cam mask
                    # Save image and heatmap
                    image_dict[args.method], _ = gen_cam(img[y1:y2, x1:x2], mask_plus_plus)
                    save_image(image_dict, os.path.basename(j), args.output, index=f'2th_stage_box{index}')


if __name__ == '__main__':
    arguments = get_parser().parse_args()
    main(arguments)
    print(f'Total training time: {datetime.now() - start}')
