from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import warnings
import cv2
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from collections import OrderedDict
from skimage import io, img_as_ubyte
from skimage.util import view_as_windows
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad

SUPPORTED_ACTIVATIONS = [
    'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus'
]

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    if op.type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """

    def __init__(self, T, X, session, keras_learning_phase=None):
        self.T = T  # target Tensor
        self.X = X  # input Tensor
        self.Y_shape = [None, ] + T.get_shape().as_list()[1:]
        # Most often T contains multiple output units. In this case, it is often necessary to select
        # a single unit to compute contributions for. This can be achieved passing 'ys' as weight for the output Tensor.
        self.Y = tf.placeholder(tf.float32, self.Y_shape)
        # placeholder_from_data(ys) if ys is not None else 1.0  # Tensor that represents weights for T
        self.T = self.T * self.Y
        self.symbolic_attribution = None
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        logging.info('Model with multiple inputs: %s' % self.has_multiple_inputs)

        # References
        self._init_references()

        # Create symbolic explanation once during construction (affects only gradient-based methods)
        self.explain_symbolic()

    def explain_symbolic(self):
        return None

    def run(self, xs, ys=None, batch_size=None):
        pass

    def _init_references(self):
        pass

    def _check_input_compatibility(self, xs, ys=None, batch_size=None):
        if ys is not None:
            if not self.has_multiple_inputs and len(xs) != len(ys):
                raise RuntimeError(
                        'When provided, ys must have the same batch size as xs (xs has batch size {} and ys {})'.format(
                                len(xs), len(ys)))
            elif self.has_multiple_inputs and np.all([len(i) != len(ys) for i in xs]):
                raise RuntimeError('When provided, ys must have the same batch size as all elements of xs')
        if batch_size is not None and batch_size > 0:
            if self.T.shape[0].value is not None and self.T.shape[0].value is not batch_size:
                raise RuntimeError('When using batch evaluation, the first dimension of the target tensor '
                                   'must be compatible with the batch size. Found %s instead' % self.T.shape[0].value)
            if isinstance(self.X, list):
                for x in self.X:
                    if x.shape[0].value is not None and x.shape[0].value is not batch_size:
                        raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                           'must be compatible with the batch size. Found %s instead' % x.shape[
                                               0].value)
            else:
                if self.X.shape[0].value is not None and self.X.shape[0].value is not batch_size:
                    raise RuntimeError('When using batch evaluation, the first dimension of the input tensor '
                                       'must be compatible with the batch size. Found %s instead' % self.X.shape[
                                           0].value)

    def _session_run_batch(self, T, xs, ys=None):
        feed_dict = {}
        if self.has_multiple_inputs:
            for k, v in zip(self.X, xs):
                feed_dict[k] = v
        else:
            feed_dict[self.X] = xs

        # If ys is not passed, produce a vector of ones that will be broadcasted to all batch samples
        feed_dict[self.Y] = ys if ys is not None else np.ones([1, ] + self.Y_shape[1:])

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(T, feed_dict)

    def _session_run(self, T, xs, ys=None, batch_size=None):
        num_samples = len(xs)
        if self.has_multiple_inputs is True:
            num_samples = len(xs[0])
            if len(xs) != len(self.X):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   % (str(len(xs)), str(len(self.X))))
            if batch_size is not None:
                for xi in xs:
                    if len(xi) != num_samples:
                        raise RuntimeError('Evaluation in batches requires all inputs to have '
                                           'the same number of samples')

        if batch_size is None or batch_size <= 0 or num_samples <= batch_size:
            return self._session_run_batch(T, xs, ys)
        else:
            outs = []
            batches = make_batches(num_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                # Get a batch from data
                xs_batch = slice_arrays(xs, batch_start, batch_end)
                # If the target tensor has one entry for each sample, we need to batch it as well
                ys_batch = None
                if ys is not None:
                    ys_batch = slice_arrays(ys, batch_start, batch_end)
                batch_outs = self._session_run_batch(T, xs_batch, ys_batch)
                batch_outs = to_list(batch_outs)
                if batch_index == 0:
                    # Pre-allocate the results arrays.
                    for batch_out in batch_outs:
                        shape = (num_samples,) + batch_out.shape[1:]
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
                for i, batch_out in enumerate(batch_outs):
                    outs[i][batch_start:batch_end] = batch_out
            return unpack_singleton(outs)


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """

    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)

    def explain_symbolic(self):
        if self.symbolic_attribution is None:
            self.symbolic_attribution = self.get_symbolic_attribution()
        return self.symbolic_attribution

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        results = self._session_run(self.explain_symbolic(), xs, ys, batch_size)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class PerturbationBasedMethod(AttributionMethod):
    """
    Base class for perturbation-based attribution methods
    """

    def __init__(self, T, X, session, keras_learning_phase):
        super(PerturbationBasedMethod, self).__init__(T, X, session, keras_learning_phase)
        self.base_activation = None

    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)

    def explain_symbolic(self):
        if self.symbolic_attribution is None:
            self.symbolic_attribution = self.get_symbolic_attribution()
        return self.symbolic_attribution

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        results = self._session_run(self.explain_symbolic(), xs, ys, batch_size)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


class DummyZero(GradientBasedMethod):
    """
    Returns zero attributions. For testing only.
    """

    def get_symbolic_attribution(self, ):
        return tf.gradients(self.T, self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)


class Saliency(GradientBasedMethod):
    """
    Saliency maps
    https://arxiv.org/abs/1312.6034
    """

    def get_symbolic_attribution(self):
        return [tf.abs(g) for g in tf.gradients(self.T, self.X)]


class GradientXInput(GradientBasedMethod):
    """
    Gradient * Input
    https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
    """

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
                tf.gradients(self.T, self.X),
                self.X if self.has_multiple_inputs else [self.X])]


class IntegratedGradients(GradientBasedMethod):
    """
    Integrated Gradients
    https://arxiv.org/pdf/1703.01365.pdf
    """

    def __init__(self, T, X, session, keras_learning_phase, steps=100, baseline=None):
        self.steps = steps
        self.baseline = baseline
        super(IntegratedGradients, self).__init__(T, X, session, keras_learning_phase)

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)

        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [b + (x - b) * alpha for x, b in zip(xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (xs - self.baseline) * alpha
            _attr = self._session_run(self.explain_symbolic(), xs_mod, ys, batch_size)
            if gradient is None:
                gradient = _attr
            else:
                gradient = [g + a for g, a in zip(gradient, _attr)]

        results = [g * (x - b) / self.steps for g, x, b in zip(
                gradient,
                xs if self.has_multiple_inputs else [xs],
                self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results


class EpsilonLRP(GradientBasedMethod):
    """
    Layer-wise Relevance Propagation with epsilon rule
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
    """

    def __init__(self, T, X, session, keras_learning_phase, epsilon=1e-4):
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon
        super(EpsilonLRP, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
                tf.gradients(self.T, self.X),
                self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + eps * tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))


class DeepLIFTRescale(GradientBasedMethod):
    """
    DeepLIFT
    This reformulation only considers the "Rescale" rule
    https://arxiv.org/abs/1704.02685
    """
    _deeplift_ref = {}

    def __init__(self, T, X, session, keras_learning_phase, baseline=None):
        self.baseline = baseline
        super(DeepLIFTRescale, self).__init__(T, X, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
                tf.gradients(self.T, self.X),
                self.X if self.has_multiple_inputs else [self.X],
                self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                        original_grad(instant_grad.op, grad))

    def _init_references(self):
        # print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)
        YR = self._session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        # print('DeepLIFT: references ready')
        sys.stdout.flush()


class Occlusion(PerturbationBasedMethod):
    """
    Occlusion method
    Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
    This method performs a systematic perturbation of contiguous hyperpatches in the input,
    replacing each patch with a user-defined value (by default 0).

    window_shape : integer or tuple of length xs_ndim
    Defines the shape of the elementary n-dimensional orthotope the rolling window view.
    If an integer is given, the shape will be a hypercube of sidelength given by its value.

    step : integer or tuple of length xs_ndim
    Indicates step size at which extraction shall be performed.
    If integer is given, then the step is uniform in all dimensions.
    """

    def __init__(self, T, X, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = X[0].get_shape().as_list()
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        logging.info('Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self, xs, ys=None, batch_size=None):
        self._check_input_compatibility(xs, ys, batch_size)
        input_shape = xs.shape[1:]
        batch_size = xs.shape[0]
        total_dim = np.asscalar(np.prod(input_shape))

        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._session_run(self.T, xs, ys, batch_size)

        # Start perturbation loop
        for i, p in enumerate(idx_patches):
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * xs
            delta = eval0 - self._session_run(self.T, masked_xs, ys, batch_size)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                          'probably because window_shape and step do not allow to cover the all input.')
        return attribution


class ShapleySampling(PerturbationBasedMethod):
    """
    Shapley Value sampling
    Computes approximate Shapley Values using "Polynomial calculation of the Shapley value based on sampling",
    Castro et al, 2009 (https://www.sciencedirect.com/science/article/pii/S0305054808000804)

    samples : integer (default 5)
    Defined the number of samples for each input feature.
    Notice that evaluating a model samples * n_input_feature times might take a while.

    sampling_dims : list of dimension indexes to run sampling on (feature dimensions).
    By default, all dimensions except the batch dimension will be sampled.
    For example, with a 4-D tensor that contains color images, single color channels are sampled.
    To sample pixels, instead, use sampling_dims=[1,2]
    """

    def __init__(self, T, X, session, keras_learning_phase, samples=5, sampling_dims=None):
        super(ShapleySampling, self).__init__(T, X, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')
        dims = len(X.shape)
        if sampling_dims is not None:
            if not 0 < len(sampling_dims) <= (dims - 1):
                raise RuntimeError('sampling_dims must be a list containing 1 to %d elements' % (dims - 1))
            if 0 in sampling_dims:
                raise RuntimeError('Cannot sample batch dimension: remove 0 from sampling_dims')
            if any([x < 1 or x > dims - 1 for x in sampling_dims]):
                raise RuntimeError('Invalid value in sampling_dims')
        else:
            sampling_dims = list(range(1, dims))

        self.samples = samples
        self.sampling_dims = sampling_dims

    def run(self, xs, ys=None, batch_size=None):
        xs_shape = list(xs.shape)
        batch_size = xs.shape[0]
        n_features = int(np.asscalar(np.prod([xs.shape[i] for i in self.sampling_dims])))
        result = np.zeros((xs_shape[0], n_features))

        run_shape = list(xs_shape)  # a copy
        run_shape = np.delete(run_shape, self.sampling_dims).tolist()
        run_shape.insert(1, -1)

        reconstruction_shape = [xs_shape[0]]
        for j in self.sampling_dims:
            reconstruction_shape.append(xs_shape[j])

        for r in range(self.samples):
            p = np.random.permutation(n_features)
            x = xs.copy().reshape(run_shape)
            y = None
            for i in p:
                if y is None:
                    y = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                x[:, i] = 0
                y0 = self._session_run(self.T, x.reshape(xs_shape), ys, batch_size)
                delta = y - y0
                delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=False)
                result[:, i] += delta_aggregated
                y = y0

        shapley = result / self.samples
        return shapley.reshape(reconstruction_shape)


class DeepExplain(object):
    def __init__(self, graph=None, session=tf.get_default_session()):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def get_explainer(self, method, T, X, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        if isinstance(X, list):
            for x in X:
                if 'tensor' not in str(type(x)).lower():
                    raise RuntimeError('If a list, X must contain only Tensorflow Tensor objects')
        else:
            if 'tensor' not in str(type(X)).lower():
                raise RuntimeError('X must be a Tensorflow Tensor object or a list of them')

        if 'tensor' not in str(type(T)).lower():
            raise RuntimeError('T must be a Tensorflow Tensor object')

        logging.info('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X,
                                       self.session,
                                       keras_learning_phase=self.keras_phase_placeholder,
                                       **kwargs)

        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return method

    def explain(self, method, T, X, xs, ys=None, batch_size=None, **kwargs):
        explainer = self.get_explainer(method, T, X, **kwargs)
        return explainer.run(xs, ys, batch_size)

    @staticmethod
    def get_override_map():
        return dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]


class GradientMethod(object):
    def __init__(self, session, image_resize, output_tensor, explainer, baseline):
        """
        Initialize GradientMethod
        :param session: Tensorflow session
        :param image_resize: Image resize
        :param output_tensor: Output tensor
        :param explainer: Explainer
        :param baseline: Baseline
        """
        self.sess = session
        self.img_rs = image_resize
        self.output_tensor = output_tensor
        self.explainer = explainer
        self.baseline = baseline

    def __call__(self, imgs, method, img_input, img_resize=None):
        """
        Calculate Gradient Method
        :param imgs: Input image
        :param method: Choose a method to calculate gradient: IntGrad, DeepLIFT or others
        :param img_resize: Resize image
        :return: Gradient Method explanation
        """
        with DeepExplain(session=self.sess) as de:
            if method in ['intgrad', 'deeplift']:
                attributions = de.explain(method,
                                          np.sum(self.output_tensor[0, :, 1:2]),
                                          self.img_rs, img_resize,
                                          baseline=self.baseline)
            else:
                img_resize = self.sess.run(self.img_rs, feed_dict={img_input: imgs})
                attributions = self.explainer.run(img_resize)
            analysis = attributions
            analysis = iutils.postprocess_images(analysis,
                                                 color_coding='BGRtoRGB',
                                                 channels_first=False)
            analysis = ivis.gamma(analysis, minamp=0, gamma=0.95)
            analysis = ivis.heatmap(analysis)
            analysis = cv2.resize(analysis[0], dsize=(imgs.shape[2], imgs.shape[1]), interpolation=cv2.INTER_LINEAR)
            return analysis


# -----------------------------------------------------------------------------
# END ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'elrp': (EpsilonLRP, 4),
    'deeplift': (DeepLIFTRescale, 5),
    'occlusion': (Occlusion, 6),
    'shapley_sampling': (ShapleySampling, 7)
})


@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]


def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.
    If a tensor is passed, we return list of size 1 containing the tensor.
    :param x: target object to be normalized.
    :param allow_tuple: If False and x is a tuple, it will be converted into a list with a single element (the tuple). Else converts the tuple to a list.
    :return: list of size 1 containing the tensor.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]


def unpack_singleton(x):
    """Gets the equivalent np-array if the iterable has only one value.
    :param x: a list of tuples.
    :return: the same iterable or the iterable converted to a np-array.
    """
    if len(x) == 1:
        return np.array(x)
    return x


def slice_arrays(arrays, start=None, stop=None):
    """
    Slices an array or list of arrays.
    :param arrays: list of arrays to slice.
    :param start: int, start index.
    :param stop: int, end index.
    :return: list of sliced arrays.
    """
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        return [None if x is None else x[start:stop] for x in arrays]
    else:
        return arrays[start:stop]


def placeholder_from_data(numpy_array):
    """
    Creates a placeholder from a numpy array.
    :param numpy_array: a numpy array.
    :return: a tensorflow placeholder.
    """
    if numpy_array is None:
        return None
    return tf.placeholder('float', [None, ] + list(numpy_array.shape[1:]))


def get_info(path):
    """
    Get the ground-truth bounding boxes and labels of the image
    :param path: Path to the xml file
    :return: list of bounding boxes and labels
    """
    gr_truth = []
    root = ET.parse(path).getroot()
    for type_tag in root.findall('object'):
        xmin = int(type_tag.find('bndbox/xmin').text)
        ymin = int(type_tag.find('bndbox/ymin').text)
        xmax = int(type_tag.find('bndbox/xmax').text)
        ymax = int(type_tag.find('bndbox/ymax').text)
        gr_truth.append([xmin, ymin, xmax, ymax])
    return gr_truth


def bb_intersection_over_union(boxA, boxB):
    """
    Calculate the intersection over union of two bounding boxes
    :param boxA: array of shape [4*1] = [x1,y1,x2,y2]
    :param boxB: array of shape [4*1] = [x1,y1,x2,y2]
    :return: IoU
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def gen_cam(image, mask, boxs=None):
    """
    Generate CAM map
    :param image: [H,W,C],the original image
    :param mask: [H,W], range 0~1
    :param boxs: [N,4], the bounding boxes
    :return: tuple(cam,heatmap)
    """
    if boxs is None:
        boxs = [[0, 0, 0, 0]]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[..., ::-1]  # gbr to rgb
    image_cam = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)

    image_cam = draw(image_cam, boxs)

    # heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    # image_cam = np.float32(image_cam) / 255
    image_cam = image_cam[..., ::-1]
    return image_cam, heatmap


def draw(image, boxs, gr_truth_boxes, threshold):
    """
    Draw bounding boxes on image
    :param image: [H,W,C],the original image
    :param boxs: [N,4], the bounding boxes
    :param gr_truth_boxes: [N,4], the ground-truth bounding boxes
    :param threshold: the threshold to filter the bounding boxes
    :return: image with bounding boxes
    """
    img_draw = image
    for a in boxs:
        iou = []
        for b in gr_truth_boxes:
            iou.append(bb_intersection_over_union(a, b))
            test_iou = any(l > threshold for l in iou)
            if test_iou:
                img_draw = cv2.rectangle(image, (a[0], a[1]), (a[2], a[3]), color=(255, 0, 0), thickness=2)
            else:
                img_draw = cv2.rectangle(image, (a[0], a[1]), (a[2], a[3]), color=(0, 0, 255), thickness=2)
    return img_draw


def save_image(image_dicts, input_image_name, output_dir, index):
    """
    Save output in folder named results
    :param image_dicts: Dictionary results
    :param input_image_name: Name of original image
    :param output_dir: Path to output directory
    :param index: Index of image
    """
    name_img = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, f'{name_img}-{key}-{index}.jpg'), img_as_ubyte(image))


def get_config(path_config):
    """
    Get config from json file
    :param path_config: Path to config file
    :return: config
    """
    with open(path_config, 'r') as fin:
        config_xAI = json.load(fin)
    return config_xAI


def get_model(model_path):
    """
    Get model from .pb file
    :param model_path: Path to .pb model file
    :return: model
    """
    global graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.GFile(model_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            tf.import_graph_def(graph_def, name='')
            img_input = graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = graph.get_tensor_by_name('detection_scores:0')
            num_detections = graph.get_tensor_by_name('num_detections:0')
            detection_classes = graph.get_tensor_by_name('detection_classes:0')

            sess = tf.Session(graph=graph)
        return sess, img_input, detection_boxes, detection_scores, num_detections, detection_classes
