import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from utils import get_center, get_tensor_mini, bbox_iou


class KDE(object):
    def __init__(self, sess, image, image_path, y_p_num_detections, y_p_boxes):
        self.image = image
        self.h_img, self.w_img = image.shape[1:3]
        self.sess = sess
        self.y_p_num_detections = y_p_num_detections
        self.y_p_boxes = y_p_boxes
        self.image_name = os.path.basename(image_path)

    def get_kde_map(self, box):
        """
        This function is to estimate the probability density function (PDF) of
        a random variable in a non-parametric way
        :return:
            kernel: KDE object
            f: KDE map add resize to the image size
        """
        x_box, y_box = get_center(box)
        x_train = np.vstack([x_box, y_box]).T
        x, y = x_train[:, 0], x_train[:, 1]
        xmin, xmax = 0, self.w_img
        ymin, ymax = 0, self.h_img
        xx, yy = np.mgrid[xmin:xmax, ymin:ymax]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        return kernel, f

    def get_kde_score(self, kde_kernel, box_predicted):
        """
        This function is to find the kde score for the box predicted by AI model
        KDE score is the ratio of the KDE value of the predicted box center
        divided by the highest KDE value on the KDE map.
        Input:
          - box_predicted: the coordinates of the predicted box
          - kde_kernel: KDE object containing the kde map that estimated from
          the previous phase
        Output:
          - predict_value_kde: KDE score of the predicted box center
        """
        x_predict, y_predict = get_center(box_predicted)
        kde_map = np.zeros((self.h_img, self.w_img))
        for i in range(self.h_img):
            for j in range(self.w_img):
                kde_map[i][j] = kde_kernel.evaluate([i, j])
        predict_value_kde = kde_kernel.evaluate([x_predict, y_predict]) / np.max(kde_map)
        return predict_value_kde

    def show_kde_map(self, box_predicted, f, save_file=None):
        """
        This function is to show the kde map in the input image
        Input:
          - box: list of coordinate of 300 boxes
          - box_predicted: coodinate of the predicted box
          - image: the predict image
        """
        xmin, xmax = 0, self.w_img
        ymin, ymax = 0, self.h_img
        xx, yy = np.mgrid[0:self.w_img, 0:self.h_img]

        fig = plt.figure()
        ax = fig.gca()
        plt.axis([xmin, xmax, ymin, ymax])
        ax.imshow(self.image[0])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ax.get_ylim()[::-1])
        x_predict, y_predict = get_center(box_predicted)
        plt.scatter(x_predict, y_predict, c='red')
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        cset = ax.contour(xx, yy, f, colors='g')
        ax.set_xlabel('Y1')
        ax.set_ylabel('Y0')
        if save_file is not None:
            plt.savefig(os.path.join(save_file, 'kde_' + self.image_name + '_kde_blue.jpg'), dpi=600)
            plt.show()

    def get_box_predicted(self, img_input):
        # Get boxes
        box = []
        box_predicted = []
        b = [n.name for n in self.sess.graph.as_graph_def().node if 'SecondStageBoxPredictor' in n.name]
        boxes = get_tensor_mini(self.sess, b[18], self.image, img_input)
        # Get predicted box
        post_process = [n.name for n in self.sess.graph.as_graph_def().node if 'SecondStagePostprocessor' in n.name]
        boxes_post_process = get_tensor_mini(self.sess, post_process[14], self.image, img_input)
        # Preprocess boxes
        ratio = self.w_img / self.h_img
        for i in range(300):
            x1, x2 = int(boxes_post_process[i][1] * ratio), int(boxes_post_process[i][3] * ratio)
            y1, y2 = int(boxes_post_process[i][0] * ratio), int(boxes_post_process[i][2] * ratio)
            box.append([x1, y1, x2, y2])

        num_box = int(self.y_p_num_detections[0])

        for i in range(num_box):
            x1, x2 = int(self.y_p_boxes[0][i][1] * self.w_img), int(self.y_p_boxes[0][i][3] * self.w_img)
            y1, y2 = int(self.y_p_boxes[0][i][0] * self.h_img), int(self.y_p_boxes[0][i][2] * self.h_img)
            box_predicted.append([x1, y1, x2, y2])

        iou = []
        for i in range(num_box):
            for j in range(300):
                iou.append(bbox_iou(box[j], box_predicted[i]))

        return box, box_predicted