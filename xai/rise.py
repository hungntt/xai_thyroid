import numpy as np
from skimage.transform import resize
from tqdm import tqdm


class RISE(object):
    def __init__(self, image, sess, grid_size, prob, num_samples=500, batch_size=1):
        """
        Initialize RISE
        :param image: Input image
        :param sess: Tensorflow session
        :param grid_size: Grid size
        :param prob: Probability of sampling
        :param num_samples: Number of samples
        :param batch_size: Batch size
        """
        self.image = image
        if num_samples > 700 or num_samples <= 0:
            num_samples = 700
        self.num_samples = num_samples
        self.sess = sess
        self.grid_size = grid_size
        self.prob = prob
        self.image_size = (image.shape[1], image.shape[2])
        self.batch_size = batch_size
        self.mask = self.generate_mask(self.num_samples, self.grid_size, self.prob)

    def generate_mask(self, num_samples, grid_size, prob):
        """
        Generate mask
        :param num_samples: Number of samples
        :param grid_size: Grid size
        :param prob: Probability of sampling
        :return: Mask
        """
        cell_size = np.ceil(np.array(self.image_size) / grid_size)
        up_size = (grid_size + 1) * cell_size
        grid = np.random.rand(num_samples, grid_size, grid_size) < prob
        grid = grid.astype('float32')
        masks = np.empty((num_samples, *self.image_size))
        for i in tqdm(range(num_samples), desc='Generating masks'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.image_size[0], y:y + self.image_size[1]]
        masks = masks.reshape(-1, 1)
        return masks

    def explain(self, image, index, mask, detection_boxes, detection_scores, num_detections, detection_classes):
        """
        Calculate RISE explanation
        :param image: Input image
        :param index: Index of image
        :param mask: Mask
        :param detection_boxes: Detection boxes
        :param detection_scores: Detection scores
        :param num_detections: Number of detections
        :param detection_classes: Detection classes
        :return: RISE explanation as saliency map
        """
        N = self.num_samples
        p = self.prob
        preds = np.array([])
        masked = self.mask * image
        for i in tqdm(range(0, N, self.batch_size)):
            input_dict = {mask: masked[i:i + self.batch_size]}
            p_boxes, p_scores, p_num_detections, p_classes = self.sess.run(
                    [detection_boxes, detection_scores, num_detections, detection_classes],
                    feed_dict=input_dict)
            for j in range(self.batch_size):
                preds = np.append(preds, p_scores[j][index])
        preds = preds.reshape(N, 1)
        sal = preds.T.dot(self.mask.reshape(N, -1))
        sal = sal.reshape(-1, *self.image_size)
        sal = sal / N / p
        sal -= np.min(sal)
        sal /= (np.max(sal) - np.min(sal))
        return sal
