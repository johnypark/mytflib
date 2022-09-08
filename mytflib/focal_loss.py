# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements Focal loss. Adapted from tensorflow-addons, fixed several things. 
    9/7/2022 Added label smoothing term
    To do list: get_config method for SigmoidFocalCrossEntropy2
    Test FC_loss3
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from typeguard import typechecked

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


class SigmoidFocalCrossEntropy2(LossFunctionWrapper):
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Usage:

    >>> fl = tfa.losses.SigmoidFocalCrossEntropy()
    >>> loss = fl(
    ...     y_true = [[1.0], [1.0], [0.0]],y_pred = [[0.97], [0.91], [0.03]])
    >>> loss
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([6.8532745e-06, 1.9097870e-04, 2.0559824e-05],
    dtype=float32)>

    Usage with `tf.keras` API:

    >>> model = tf.keras.Model()
    >>> model.compile('sgd', loss=tfa.losses.SigmoidFocalCrossEntropy())

    Args:
      alpha: balancing factor, default value is 0.25.
      gamma: modulating factor, default value is 2.0.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
          shape as `y_true`; otherwise, it is scalar.

    Raises:
        ValueError: If the shape of `sample_weight` is invalid or value of
          `gamma` is less than zero.
    """

    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "sigmoid_focal_crossentropy",
        label_smoothing: FloatTensorLike = 0,
        **kwargs
    ):
        
        super().__init__(
            sigmoid_focal_crossentropy2,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
            label_smoothing = label_smoothing
        )


def sigmoid_focal_crossentropy2(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False,
    label_smoothing: FloatTensorLike = 0.0
) -> tf.Tensor:
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.

    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)  
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    def _smooth_labels_(y_true, label_smoothing):
        y_true = y_true*(1 - label_smoothing) + label_smoothing/(num_classes) + \
        (1 - y_true)*(label_smoothing/(num_classes))
        return y_true
    
    y_true = _smooth_labels_(y_true, label_smoothing = label_smoothing)
    
    #tf.__internal__.smart_cond.smart_cond(label_smoothing,
             #                                       _smooth_labels, lambda: y_true)
    
    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits = from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0
    
    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)/(num_classes-1) # if not binary, need to change to alpha factor to account for num of classes
        # essentially, negative samples weight gets down with the magnitude of the classes

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        focal_weight = K.clip((1.0 - p_t), K.epsilon(), 1.0) 
        modulating_factor = tf.pow(focal_weight, gamma)
        
    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

#https://github.com/keras-team/keras-cv/blob/master/keras_cv/losses/focal.py
class FC_loss3(tf.keras.losses.Loss): # rewrote keras_cv focal loss, since it is not correctly written for label smoothing,
    # and it generates NaN when gamma = 0.5.
    """Implements Focal loss
    Focal loss is a modified cross-entropy designed to perform better with
    class imbalance. For this reason, it's commonly used with object detectors.
    Args:
        alpha: a float value between 0 and 1 representing a weighting factor
            used to deal with class imbalance. Positive classes and negative
            classes have alpha and (1 - alpha) as their weighting factors
            respectively. Defaults to 0.25.
        gamma: a positive float value representing the tunable focusing
            parameter. Defaults to 2.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, `y_pred` is assumed to encode a probability distribution.
            Default to `False`.
        label_smoothing: Float in `[0, 1]`. If higher than 0 then smooth the
            labels by squeezing them towards `0.5`, i.e., using `1. - 0.5 * label_smoothing`
            for the target class and `0.5 * label_smoothing` for the non-target
            class. THIS PART NEEDS REWRITING 
    References:
        - [Focal Loss paper](https://arxiv.org/abs/1708.02002)
    Standalone usage:
    ```python
    y_true = tf.random.uniform([10], 0, maxval=4)
    y_pred = tf.random.uniform([10], 0, maxval=4)
    loss = FocalLoss()
    loss(y_true, y_pred).numpy()
    ```
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.FocalLoss())
    ```
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        from_logits=False,
        label_smoothing=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    
    def _smooth_labels(self, y_true, label_smoothing, num_classes):
        
        y_true = y_true*(1 - label_smoothing) + label_smoothing/(num_classes) + \
        (1 - y_true)*(label_smoothing/(num_classes))
        return y_true

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
    
        if self.label_smoothing:
            y_true = self._smooth_labels(y_true, 
                                         label_smoothing =  tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype),
                                         num_classes = num_classes)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        cross_entropy = K.binary_crossentropy(y_true, y_pred)

        alpha = y_true * self._alpha + (1 - y_true) * (1 - self._alpha)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = K.clip((1.0 - p_t), K.epsilon(), 1.0) 
        loss = alpha * tf.pow(focal_weight, self._gamma) * cross_entropy
        # In most losses you mean over the final axis to achieve a scalar
        # Focal loss however is a special case in that it is meant to focus on
        # a small number of hard examples in a batch.  Most of the time this
        # comes in the form of thousands of background class boxes and a few
        # positive boxes.
        # If you mean over the final axis you will get a number close to 0,
        # which will encourage your model to exclusively predict background
        # class boxes.
        return K.sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config