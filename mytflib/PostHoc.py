import tensorflow as tf


class GradCAM:
    """
    Grad-CAM: https://arxiv.org/pdf/1611.07450.pdf

    """

    def __init__(self, grad_model):
      self.grad_model = grad_model

    def __call__(self, input):
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(input)
            score = tf.reduce_max(preds, axis = 1)
        grads = tape.gradient(score, last_conv_layer_output)      
        cam = tf.reduce_sum(grads * last_conv_layer_output, 3)
        cam = tf.nn.relu(cam)
        cam = cam / tf.reduce_max(cam, axis = (1,2), keepdims = True)
        return cam
