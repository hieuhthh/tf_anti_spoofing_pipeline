import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

from augment import *

def get_data_from_phrase(route, phrase=None):
    """
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """
    X_path = []
    Y_int = []
    phrase_path = os.path.join(route, phrase) if phrase is not None else route
    all_class = sorted(os.listdir(phrase_path))
    for cl in all_class:
        path2cl = os.path.join(phrase_path, cl)
        temp =  glob(path2cl + '/*')
        X_path = X_path + temp
        Y_int = Y_int + len(temp) * [all_class.index(cl)]
    return X_path, Y_int, all_class

def auto_split_data(route, fold, valid_fold=None, test_fold=None, stratified=False, seed=42):
    """
    input:
        route to the directory that its subfolder are classes
        fold: how many folds
    output:
        X, Y, class corresponding
    """
    X_path, Y_int, all_class = get_data_from_phrase(route)

    df = pd.DataFrame({'image':X_path, 'label':Y_int})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if stratified:
        skf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
        for idx, (train_index, test_index) in enumerate(skf.split(df.image, df.label)):
            df.loc[test_index, 'fold'] = idx
    else:
        kf = KFold(n_splits=fold, random_state=seed, shuffle=True)
        for idx, (train_index, test_index) in enumerate(kf.split(df.image)):
            df.loc[test_index, 'fold'] = idx

    if valid_fold is not None and test_fold is not None:
        df_train = df[df['fold']!=valid_fold]
        df_train = df_train[df_train['fold']!=test_fold]

        df_valid = df[df['fold']==valid_fold]
        df_test = df[df['fold']==test_fold]

        X_train = df_train['image'].values
        X_valid = df_valid['image'].values
        X_test = df_test['image'].values

        Y_train = df_train['label'].values
        Y_valid = df_valid['label'].values
        Y_test = df_test['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid, X_test, Y_test

    elif valid_fold is not None:
        df_train = df[df['fold']!=valid_fold]

        df_valid = df[df['fold']==valid_fold]

        X_train = df_train['image'].values
        X_valid = df_valid['image'].values

        Y_train = df_train['label'].values
        Y_valid = df_valid['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid

    else:
        df_train = df

        X_train = df_train['image'].values

        Y_train = df_train['label'].values

        return X_train, Y_train, all_class

def build_decoder(with_labels=True, label_mode='int', all_class=None, target_size=(256, 256), im_size_before_crop=None):
    def decode_img_preprocess(img):
        if im_size_before_crop is None:
            img = tf.image.resize(img, target_size)
        else:
            img = tf.image.resize(img, (im_size_before_crop, im_size_before_crop))
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_img(path):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img = decode_img_preprocess(img)
        return img
    
    def decode_label(label):
        """
        label: int label
        """
        if label_mode == 'int':
            return label
        if label_mode == 'cate': # categorical
            return tf.one_hot(label, len(all_class))
        if label_mode == 'cate_int':
            return tf.one_hot(label, len(all_class)), label
        raise ValueError("Label mode not supported")
        
    def decode_with_labels(path, label):
        return decode_img(path), decode_label(label)
    
    return decode_with_labels if with_labels else decode_img

def build_dataset(paths, labels=None, bsize=32,
                  decode_fn=None, augment=None,
                  repeat=False, shuffle=1024,
                  cache=False, cache_dir=""):
    """
    paths: paths to images
    labels: int label
    """              
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((paths))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    # dset = dset.map(augment, num_parallel_calls=AUTO) if augment is not None else dset
    dset = dset.map(lambda x,y:(augment(x),y), num_parallel_calls=AUTO) if augment is not None else dset
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

def build_dataset_from_X_Y(X_path, Y_int, all_class, with_labels, label_mode, img_size,
                           batch_size, repeat, shuffle, augment, im_size_before_crop=None):
    decoder = build_decoder(with_labels=with_labels, label_mode=label_mode, all_class=all_class, 
                            target_size=img_size, im_size_before_crop=im_size_before_crop)

    augment_img = build_augment() if augment else None

    dataset = build_dataset(X_path, Y_int, bsize=batch_size, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle, augment=augment_img)

    return dataset

if __name__ == '__main__':
    from utils import *
    from multiprocess_dataset import *

    settings = get_settings()
    globals().update(settings)

    route_dataset = path_join(route, 'dataset')

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    use_cate_int = False
    if label_mode == 'cate_int':
        use_cate_int = True

    X_train, Y_train, all_class, X_valid, Y_valid = auto_split_data_multiprocessing_faster(route_dataset, valid_ratio, test_ratio, seed)
    
    train_n_images = len(Y_train)
    train_dataset = build_dataset_from_X_Y(X_train, Y_train, all_class, train_with_labels, label_mode, img_size,
                                           BATCH_SIZE, train_repeat, train_shuffle, train_augment, im_size_before_crop)

    valid_n_images = len(Y_valid)
    valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, all_class, valid_with_labels, label_mode, img_size,
                                           BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

    print(len(all_class))
    print(len(X_train))
    print(len(X_valid))
    print(X_train[0])
    print(Y_train[0])

    for x, y in train_dataset:
        break
    print(x)
    print(y)

    import cv2
    import numpy as np
    cv2.imwrite("sample.png", np.array(x[0][...,::-1])*255)

