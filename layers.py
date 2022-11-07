import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU, Reshape, \
DepthwiseConv2D, Multiply, Add

class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, append_norm=False, partial_fc_split=0, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        # self.init = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
        self.init = keras.initializers.glorot_normal()
        # self.init = keras.initializers.TruncatedNormal(mean=0, stddev=0.01)
        self.units, self.loss_top_k, self.append_norm, self.partial_fc_split = units, loss_top_k, append_norm, partial_fc_split
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = False

    def build(self, input_shape):
        if self.partial_fc_split > 1:
            self.cur_id = self.add_weight(name="cur_id", shape=(), initializer="zeros", dtype="int64", trainable=False)
            self.sub_weights = self.add_weight(
                name="norm_dense_w_subs",
                shape=(self.partial_fc_split, input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        else:
            self.w = self.add_weight(
                name="norm_dense_w",
                shape=(input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        if self.partial_fc_split > 1:
            # self.sub_weights.scatter_nd_update([[(self.cur_id - 1) % self.partial_fc_split]], [self.w])
            # self.w.assign(tf.gather(self.sub_weights, self.cur_id))
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-5)
        output = K.dot(norm_inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "append_norm": self.append_norm,
                "partial_fc_split": self.partial_fc_split,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def conv(inputs, filters, kernel_size=3, strides=(1, 1), padding='same', activation=None):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(inputs)
    return x

def bn_act(inputs, activation='swish'):
    x = BatchNormalization()(inputs)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation='swish'):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    return x

def block_multi_kernel_size(inputs, filters, kernel_sizes):
    xlist = []
    for kernel_size in kernel_sizes:
        x = conv(inputs, filters, kernel_size)
        xlist.append(x)
    x = Concatenate(axis=3)(xlist)
    x = conv_bn_act(x, filters, 1)
    return x

def self_attention(inputs, filters):
    x = conv_bn_act(inputs, filters, 1, activation=None)
    norm_x = tf.math.l2_normalize(x)
    x = tf.keras.activations.relu(x)
    x = conv_bn_act(x, filters, 1, activation=None)
    x = tf.keras.activations.softplus(x)
    x = x * norm_x
    return x

def local_branch(inputs, filters, kernel_sizes):
    x = block_multi_kernel_size(inputs, filters, kernel_sizes)
    x = self_attention(x, filters)
    return x

def orthogonal_fusion(fl, fg):
    """
    fl: local (h1, w1, c1)
    fg: global (c2)
    c1 == c2
    Return:
    (h1, w1, c1 + c2)
    """
    fl = tf.transpose(fl, [0,3,1,2])
    bs, c, w, h = fl.shape
    fl_b = tf.reshape(fl, [tf.shape(fl)[0],c,w*h])
    fl_dot_fg = tf.matmul(fg[:,tf.newaxis,:] ,fl_b)
    fl_dot_fg = tf.reshape(fl_dot_fg, [tf.shape(fl_dot_fg)[0],1,w,h])
    fg_norm = tf.norm(fg, ord=2, axis=1)
    fl_proj = (fl_dot_fg / fg_norm[:,tf.newaxis,tf.newaxis,tf.newaxis]) * fg[:,:,tf.newaxis,tf.newaxis]
    fl_orth = fl - fl_proj
    fg_rep = tf.tile(fg[:,:,tf.newaxis,tf.newaxis], multiples=(1,1,w,h))
    f_fused = tf.keras.layers.Concatenate(axis=1)([fl_orth, fg_rep])
    f_fused = tf.transpose(f_fused, [0,2,3,1])
    return f_fused

def se_module(inputs, se_ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    reduction = filters // se_ratio
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = Conv2D(reduction, kernel_size=1, use_bias=True)(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True)(se)
    se = Activation("sigmoid")(se)
    
    return Multiply()([inputs, se])

def mb_conv(inputs, filters, stride, expand_ratio, kernel_size=3, drop_rate=0, use_se=0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    
    is_fused = True if use_se == 0 else False
    shortcut = True if filters == input_channel and stride == 1 else False
        
    if is_fused and expand_ratio != 1:
        x = conv(inputs, input_channel * expand_ratio, 3, stride)
        x = bn_act(x)
    elif expand_ratio != 1: # not is fused
        x = conv(inputs, input_channel * expand_ratio, 1, 1)
        x = bn_act(x)
    else:
        x = inputs
        
    if not is_fused:
        x = DepthwiseConv2D(kernel_size, padding="same", strides=stride, use_bias=False)(x)
        x = bn_act(x)
        
    if use_se:
        x = se_module(x, se_ratio=expand_ratio)
    
    # pw-linear
    if is_fused and expand_ratio == 1:
        x = conv(x, filters, 3, strides=stride)
        x = bn_act(x)
    else:
        x = conv(x, filters, 1, strides=(1, 1))
        x = bn_act(x)
        
    if shortcut:
        if drop_rate > 0:
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
        return Add()([inputs, x])
    else:
        return x

class wBiFPNAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=(num_in,),
                                initializer=tf.keras.initializers.constant(1 / num_in),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
          'epsilon': self.epsilon
        })
        return config