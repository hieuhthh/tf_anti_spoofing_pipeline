import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, Optimizer
import tensorflow_addons as tfa

class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(tf.math.not_equal(y_true, 0))
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
 
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = y_pred_vals
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3

        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)

        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale

        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLoss, self).get_config()
        config.update(
            {
                "margin1": self.margin1,
                "margin2": self.margin2,
                "margin3": self.margin3,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)