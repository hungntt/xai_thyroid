import numpy as np
import cv2

from utils import get_tensor_mini


class DensityMap(object):
    def __init__(self, sess, image):
        self.sess = sess
        self.image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).reshape((1, image.shape[0], image.shape[1], 3))

    def explain(self, img_input, y_p_num_detections, y_p_boxes):
        b = [n.name for n in self.sess.graph.as_graph_def().node if 'SecondStageBoxPredictor' in n.name]
        post_process = [n.name for n in self.sess.graph.as_graph_def().node if 'SecondStagePostprocessor' in n.name]

        boxes = get_tensor_mini(self.sess, b[18], self.image, img_input)
        boxes_post_process = get_tensor_mini(self.sess, post_process[14], self.image, img_input)

        box = []
        h_img, w_img = self.image.shape[:2]
        ratio = w_img / h_img

        num_box = int(y_p_num_detections[0])
        box_predicted = []
        for i in range(num_box):
            x1, x2 = int(y_p_boxes[0][i][1] * w_img), int(y_p_boxes[0][i][3] * w_img)
            y1, y2 = int(y_p_boxes[0][i][0] * h_img), int(y_p_boxes[0][i][2] * h_img)
            box_predicted.append([x1, y1, x2, y2])

        for i in range(300):
            x1, x2 = int(boxes_post_process[i][1] * ratio), int(boxes_post_process[i][3] * ratio)
            y1, y2 = int(boxes_post_process[i][0] * ratio), int(boxes_post_process[i][2] * ratio)
            box.append([x1, y1, x2, y2])

        h_image, w_image = self.image.shape[0], self.image.shape[1]
        num_box = len(box)
        density_map = np.zeros((h_image, w_image))
        for i in range(num_box):
            box_a = box[i]
            for j in range(max(box_a[1], 0), min(box_a[3], h_image)):
                for k in range(max(box_a[0], 0), min(box_a[2], w_image)):
                    density_map[j][k] += 1
        max_density = np.max(density_map)
        density_map_normalize = (density_map * 256 / max_density).astype(np.uint8)
        heatmap = cv2.applyColorMap(density_map_normalize, cv2.COLORMAP_JET)

        return heatmap
