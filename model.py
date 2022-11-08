from layers import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext

def get_base_model(name, input_shape):
    if name == 'EfficientNetV2S':
        return efficientnet.EfficientNetV2S(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV1B1':
        return efficientnet.EfficientNetV1B1(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B2':
        return efficientnet.EfficientNetV1B2(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B3':
        return efficientnet.EfficientNetV1B3(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B4':
        return efficientnet.EfficientNetV1B4(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B5':
        return efficientnet.EfficientNetV1B5(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B6':
        return efficientnet.EfficientNetV1B6(num_classes=0, input_shape=input_shape, pretrained="noisy_student")

    if name == 'EfficientNetV1B7':
        return efficientnet.EfficientNetV1B7(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'ConvNeXtTiny':
        return convnext.ConvNeXtTiny(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    raise Exception("Cannot find this base model:", name)

def create_emb_model(base, final_dropout=0.1, have_emb_layer=True, emb_dim=128, name="embedding"):
    feature = base.output

    x = GlobalAveragePooling2D()(feature)
    x = Dropout(final_dropout)(x)

    if have_emb_layer:
        x = Dense(emb_dim, use_bias=False, name='bottleneck')(x)
        x = BatchNormalization(name='bottleneck_bn')(x)
    
    model = Model(base.input, x, name=name)

    return model

def create_model(max_frames, input_shape, emb_model, emb_dim, final_dropout, n_labels, 
                 trans_layers, num_heads, mlp_dim,
                 use_normdense=False, use_cate_int=False):
    input_time_shape = (max_frames, *input_shape)
    
    inp = Input(shape=input_time_shape, name="input_1")

    x = Lambda(lambda x: tf.reshape(x, [-1, *input_shape]))(inp)

    x = emb_model(x)

    x = Lambda(lambda x: tf.reshape(x, [-1, max_frames, emb_dim]))(x)

    pe = PositionEmbedding(input_shape=(max_frames, emb_dim),
                           input_dim=max_frames,
                           output_dim=emb_dim,
                           mode=PositionEmbedding.MODE_ADD,
    )

    x = pe(x)

    for i in range(trans_layers):
        x = TransformerEncoder(emb_dim, mlp_dim, num_heads)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)

    x = Dense(emb_dim // 2, use_bias=False, name='bottleneck')(x)
    x = BatchNormalization(name='bottleneck_bn')(x)
    x = Dropout(0.1)(x)

    if use_normdense:
        out = NormDense(n_labels, activation='softmax', name='cate_output')(x)
    else:
        out = Dense(n_labels, activation='softmax', name='cate_output')(x)

    if not use_cate_int:
        model = Model([inp], [out])
    else:
        model = Model([inp], [out, x])
    
    return model

if __name__ == "__main__":
    import os
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    use_cate_int = False
    if label_mode == 'cate_int':
        use_cate_int = True

    n_labels = 2

    base = get_base_model(base_name, input_shape)
    emb_model = create_emb_model(base, final_dropout, have_emb_layer, emb_dim)
    model = create_model(max_frames, input_shape, emb_model, emb_dim, final_dropout, n_labels, 
                         trans_layers, num_heads, mlp_dim,
                         use_normdense, use_cate_int)

    model.summary()

    inp = tf.ones((BATCH_SIZE, max_frames, im_size, im_size, 3))
    out = model(inp)
    print('out', out)