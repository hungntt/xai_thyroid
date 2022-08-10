import cv2
import numpy as np
import tensorflow as tf

from utils import softmax


class AdaSISE(object):
    def __init__(self, image, sess):
        self.image = image
        self.sess = sess
        self.tensor_names = [n.name for n in self.sess.graph.as_graph_def().node]
        self.last_block = [n.name for n in self.sess.graph.as_graph_def().node if 'Pool' in n.name]
        self.target_layer = self.sess.graph.get_tensor_by_name('Softmax:0')

    def explain(self, img, img_input):
        feed_dict = {img_input: self.image}

        l1 = self.sess.graph.get_tensor_by_name(
                'FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu:0')
        l2 = self.sess.graph.get_tensor_by_name(
                'FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu:0')
        l3 = self.sess.graph.get_tensor_by_name(
                'FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Mixed_5b/Branch_2/Conv2d_0c_3x3/Relu:0')
        l4 = self.sess.graph.get_tensor_by_name(
                'FirstStageFeatureExtractor/InceptionResnetV2/InceptionResnetV2/Mixed_6a/Branch_1/Conv2d_1a_3x3/Relu:0')
        l5 = self.sess.graph.get_tensor_by_name('Conv/Relu6:0')

        grad1 = tf.gradients(np.sum(self.target_layer[0, :, 1:2]), l1)[0]
        grad2 = tf.gradients(np.sum(self.target_layer[0, :, 1:2]), l2)[0]
        grad3 = tf.gradients(np.sum(self.target_layer[0, :, 1:2]), l3)[0]
        grad4 = tf.gradients(np.sum(self.target_layer[0, :, 1:2]), l4)[0]
        grad5 = tf.gradients(np.sum(self.target_layer[0, :, 1:2]), l5)[0]

        o1, g1 = self.sess.run([l1, grad1], feed_dict=feed_dict)
        o2, g2 = self.sess.run([l2, grad2], feed_dict=feed_dict)
        o3, g3 = self.sess.run([l3, grad3], feed_dict=feed_dict)
        o4, g4 = self.sess.run([l4, grad4], feed_dict=feed_dict)
        o5, g5 = self.sess.run([l5, grad5], feed_dict=feed_dict)

        a1 = np.mean(g1[0], axis=(0, 1))
        a2 = np.mean(g2[0], axis=(0, 1))
        a3 = np.mean(g3[0], axis=(0, 1))
        a4 = np.mean(g4[0], axis=(0, 1))
        a5 = np.mean(g5[0], axis=(0, 1))

        a1_nor = a1 / a1.max()
        a2_nor = a2 / a2.max()
        a3_nor = a3 / a3.max()
        a4_nor = a4 / a4.max()
        a5_nor = a5 / a5.max()

        b1 = np.where(a1_nor > 0)[0]
        b2 = np.where(a2_nor > 0)[0]
        b3 = np.where(a3_nor > 0)[0]
        b4 = np.where(a4_nor > 0)[0]
        b5 = np.where(a5_nor > 0)[0]

        a1_nor_p = a1_nor[b1]
        a2_nor_p = a2_nor[b2]
        a3_nor_p = a3_nor[b3]
        a4_nor_p = a4_nor[b4]
        a5_nor_p = a5_nor[b5]

        th1 = cv2.threshold(
                (a1_nor_p * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0] / 255

        th2 = cv2.threshold(
                (a2_nor_p * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0] / 255

        th3 = cv2.threshold(
                (a3_nor_p * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0] / 255

        th4 = cv2.threshold(
                (a4_nor_p * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0] / 255

        th5 = cv2.threshold(
                (a5_nor_p * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0] / 255

        b1b = np.where(a1_nor > th1)[0]
        b2b = np.where(a2_nor > th2)[0]
        b3b = np.where(a3_nor > th3)[0]
        b4b = np.where(a4_nor > th4)[0]
        b5b = np.where(a5_nor > th5)[0]

        score_saliency_map = []

        o = [o1, o2, o3, o4, o5]
        b = [b1b, b2b, b3b, b4b, b5b]

        for k in range(len(o)):
            score_saliency = 0
            for j in (b[k]):
                md1 = cv2.resize(o[k][0, :, :, j], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                if md1.max() == md1.min():
                    continue
                md1 = (md1 - np.min(md1)) / (np.max(md1) - np.min(md1))
                img_md1 = ((img * (md1[:, :, None].astype(np.float32))).astype(np.uint8))
                img_md1 = img_md1[None, :]
                output_md1 = self.sess.run(self.target_layer, feed_dict={img_input: img_md1})
                x = softmax(output_md1[0, :])
                score = np.sum(x[:, 1])
                score_saliency += score * md1
            score_saliency_map.append(score_saliency)
        for i in range(len(score_saliency_map)):
            if type(score_saliency_map[i]) == int:
                continue
            score_saliency_map[i] = (score_saliency_map[i] - score_saliency_map[i].min()) / (
                    score_saliency_map[i].max() - score_saliency_map[i].min())

        s0 = np.array(score_saliency_map[0] * 255, dtype=np.uint8)
        s1 = np.array(score_saliency_map[1] * 255, dtype=np.uint8)
        s2 = np.array(score_saliency_map[2] * 255, dtype=np.uint8)
        s3 = np.array(score_saliency_map[3] * 255, dtype=np.uint8)
        s4 = np.array(score_saliency_map[4] * 255, dtype=np.uint8)

        block2 = cv2.threshold(s1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1] / 255
        block3 = cv2.threshold(s2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1] / 255
        block4 = cv2.threshold(s3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1] / 255
        block5 = cv2.threshold(s4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )[1] / 255

        score_1_2 = score_saliency_map[0] + score_saliency_map[1]
        block_1_2 = score_1_2 * block2
        block_1_2 = block_1_2
        score_1_2_3 = block_1_2 + score_saliency_map[2]
        block_1_2_3 = score_1_2_3 * block3
        block_1_2_3 = block_1_2_3
        score_1_2_3_4 = block_1_2_3 + score_saliency_map[3]
        block_1_2_3_4 = score_1_2_3_4 * block4
        block_1_2_3_4 = block_1_2_3_4
        score_1_2_3_4_5 = block_1_2_3_4 + score_saliency_map[4]
        block_1_2_3_4_5 = score_1_2_3_4_5 * block5
        block_1_2_3_4_5 = block_1_2_3_4_5
        block_1_2_3_4_5 -= block_1_2_3_4_5.min()
        block_1_2_3_4_5 /= (block_1_2_3_4_5.max() - block_1_2_3_4_5.min())

        cam_resized = cv2.resize(block_1_2_3_4_5, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        image_cam = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
        image_cam = image_cam[..., ::-1]

        return image_cam
