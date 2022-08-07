import numpy as np
import skimage

from skimage.segmentation import slic
from lime import lime_image


class LIME(object):
    def __init__(self, session, image, indices, top_labels=1, num_features=60):
        """
        Initialize LIME
        :param session: Tensorflow session
        :param image: Input image
        :param indices: Indices of image
        :param top_labels: Number of top labels
        :param num_features: Number of features for segmentation
        """
        self.image = image
        self.sess = session
        self.top_labels = top_labels
        self.result = None
        self.num_features = num_features
        self.indices = indices
        self.segments = self.segment_fn(self.image)

    def segment_fn(self, image):
        """
        Segment image
        :param image: Input image
        :return: Segmented image
        """
        segments_slic = slic(image, n_segments=self.num_features, compactness=30, sigma=3)
        return segments_slic

    def _predict_(self, sample, img_input, detection_boxes, detection_scores, num_detections, detection_classes,
                  flag=False):
        """
        Predict image
        :param sample: Input image
        :param flag: Flag for prediction
        :return: Prediction
        """
        img = sample
        input_dict = {img_input: img}
        p_boxes, p_scores, p_num_detections, p_classes = self.sess.run(
                [detection_boxes, detection_scores, num_detections, detection_classes],
                feed_dict=input_dict)
        if flag:
            return p_boxes, p_scores, p_num_detections, p_classes
        else:
            rs = np.array([])
            n = img.shape[0]
            for i in range(n):
                rs = np.append(rs, p_scores[i][self.indices])
            return rs.reshape(n, 1)

    def explain(self, num_features, num_samples=100, top_labels=0, positive=False):
        """
        Calculate LIME explanation
        :param num_features: Number of features for segmentation
        :param num_samples: Number of samples
        :param top_labels: Number of top labels
        :param positive: Flag for positive explanation
        :return: LIME explanation
        """
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(self.image,
                                                 self._predict_,
                                                 num_samples=num_samples,
                                                 top_labels=self.top_labels,
                                                 hide_color=0,
                                                 segmentation_fn=self.segment_fn)
        self.result = explanation
        temp, mask = self.result.get_image_and_mask(self.result.top_labels[top_labels],
                                                    positive_only=positive,
                                                    num_features=num_features,
                                                    hide_rest=False,
                                                    min_weight=0.)
        img_boundary = skimage.segmentation.mark_boundaries(temp, mask)
        return img_boundary
