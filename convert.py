import os

from utils import *
from layers import *
from model import *
from losses import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
strategy = tf.device('/cpu:0')

des = path_join(route, 'weights')
mkdir(des)

model_weight = path_join(route, 'best_model_EfficientNetV2S_160_512_364103.h5')
encoder_name = 'encoder_v' + str(len(os.listdir(des)) + 1) + '_' + model_weight.split('/')[-1]
encoder_save = path_join(des, encoder_name)

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

n_labels = 364103
use_cate_int = True

with strategy:
    base = get_base_model(base_name, input_shape)
    emb_model = create_emb_model(base, final_dropout, have_emb_layer, emb_dim)
    model = create_model(input_shape, emb_model, n_labels, use_normdense, use_cate_int)
    model.load_weights(model_weight)
    model.summary()

with strategy:
    encoder = tf.keras.Sequential([
        model.get_layer('input_1'),
        model.get_layer('embedding'),
    ])
    encoder.summary()
    encoder.save(encoder_save)

    # encoder = tf.keras.models.load_model(encoder_save)
    # encoder.summary()