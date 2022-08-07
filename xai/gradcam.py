# --------------------------------GradCAM---------------------------------------
import cv2
import numpy as np


class GradCAM(object):
    """
    1: GradCAM calculate gradient on two stages
    2: Output tensor: Prediction boxes before Non-Max Suppression (first_stage)
    3: Get index target boxes to backpropagation (second_stage), output: The final prediction of the model
    """

    def __init__(self, session, conv_tensor, output_tensor):
        """
        Initialize GradCAM
        :param session: Tensorflow session
        :param conv_tensor: Tensor of convolution layer
        :param output_tensor: Tensor of output layer
        """
        self.sess = session
        self.conv_tensor = conv_tensor
        self.output_tensor = output_tensor

    def __call__(self, imgs, grads, img_input, stage, y_p_boxes, indices=0, index=0):
        """
        Calculate GradCAM
        :param imgs: Input image
        :param grads: Gradient of output layer

        :param stage: Choose a stage to visualize: first_stage or second_stage
        :param indices: Index of target boxes to backpropagation (second_stage)
        :param index: Index of image
        :return: GradCAM explanation
        """
        if stage == 'first_stage':
            # first image in batch
            conv_output, grads_val = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            weights = np.mean(grads_val[indices], axis=(0, 1))
            feature = conv_output[indices]
            cam = feature * weights[np.newaxis, np.newaxis, :]
        else:
            conv_output, grads_val = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            weights = np.mean(grads_val[indices], axis=(0, 1))
            feature = conv_output[indices]
            cam = feature * weights[np.newaxis, np.newaxis, :]
        cam = np.sum(cam, axis=2)
        # cam = np.maximum(cam, 0) #Relu
        # Normalize data (0, 1)
        cam -= np.min(cam)
        cam /= (np.max(cam) - np.min(cam))
        h_img, w_img = imgs.shape[1:3]
        x1, x2 = int(y_p_boxes[0][index][1] * w_img), int(y_p_boxes[0][index][3] * w_img)
        y1, y2 = int(y_p_boxes[0][index][0] * h_img), int(y_p_boxes[0][index][2] * h_img)
        # Resize CAM
        if stage == 'first_stage':
            cam = cv2.resize(cam, (w_img, h_img))
            return cam
        else:
            cam = cv2.resize(cam, (x2 - x1, y2 - y1))
            return cam, x1, y1, x2, y2


# --------------------------------GradCAM++-------------------------------------

class GradCAMPlusPlus(GradCAM):
    def __init__(self, session, conv_tensor, output_tensor):
        """
        Initialize GradCAM++
        :param session: Tensorflow session
        """
        super().__init__(session, conv_tensor, output_tensor)

    def __call__(self, imgs, grads, img_input, stage, y_p_boxes, indices=0, index=0):
        """
        Calculate GradCAM++
        :param imgs: Input image
        :param grads: Gradient of output layer
        :param stage: Choose a stage to visualize: first_stage or second_stage
        :param indices: Index of target boxes to backpropagation (second_stage)
        :param index: Index of image
        :return: GradCAM++ explanation
        """
        if stage == 'first_stage':
            outputs, grads_val_1 = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            grads_val_2 = grads_val_1 ** 2
            grads_val_3 = grads_val_2 * grads_val_1
            global_sum = np.sum(outputs[0], axis=(0, 1))
            eps = 0.000001
            aij = grads_val_2[indices] / (
                    2 * grads_val_2[indices] + global_sum[None, None, :] * grads_val_3[indices] + eps)
            aij = np.where(grads_val_1[indices] != 0, aij, 0)
            weights = np.maximum(grads_val_1[indices], 0) * aij  # Relu * aij = weight
            weights = np.sum(weights, axis=(0, 1))
            cam = outputs[indices] * weights[np.newaxis, np.newaxis, :]
        else:
            outputs, grads_val_1 = self.sess.run([self.conv_tensor, grads], feed_dict={img_input: imgs})
            grads_val_2 = grads_val_1 ** 2
            grads_val_3 = grads_val_2 * grads_val_1
            global_sum = np.sum(outputs[0], axis=(0, 1))
            eps = 0.000001
            aij = grads_val_2[indices] / (
                    2 * grads_val_2[indices] + global_sum[None, None, :] * grads_val_3[indices] + eps)
            aij = np.where(grads_val_1[indices] != 0, aij, 0)
            weights = np.maximum(grads_val_1[indices], 0) * aij  # Relu * aij = weight
            weights = np.sum(weights, axis=(0, 1))
            cam = outputs[indices] * weights[np.newaxis, np.newaxis, :]
        cam = np.sum(cam, axis=2)
        # cam = np.maximum(cam, 0) #Relu
        # Normalize
        cam -= np.min(cam)
        cam /= (np.max(cam) - np.min(cam))
        h_img, w_img = imgs.shape[1:3]
        x1, x2 = int(y_p_boxes[0][index][1] * w_img), int(y_p_boxes[0][index][3] * w_img)
        y1, y2 = int(y_p_boxes[0][index][0] * h_img), int(y_p_boxes[0][index][2] * h_img)
        # Resize cam

        if stage == 'first_stage':
            cam = cv2.resize(cam, (w_img, h_img))
            return cam
        else:
            cam = cv2.resize(cam, (x2 - x1, y2 - y1))
            return cam, x1, y1, x2, y2
