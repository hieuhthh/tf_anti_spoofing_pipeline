import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
import multiprocessing

def get_data_from_phrase_multiprocessing(route, phrase=None):
    """
    using multiprocessing
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """

    global task

    def task(phrase_path, all_class, list_cls):
        X = []
        Y = []

        for cl in list_cls:
            path2cl = os.path.join(phrase_path, cl)
            temp = glob(path2cl + '/*')
            X += temp
            Y += len(temp) * [all_class.index(cl)]

        return X, Y

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    processes = []

    phrase_path = os.path.join(route, phrase) if phrase is not None else route
    all_class = sorted(os.listdir(phrase_path))
    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    for i in range(cpu_count):
        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task, args=(phrase_path,all_class,list_cls,))
        processes.append(p)

    result = [p.get() for p in processes]

    X_path = []
    Y_int = []

    for x, y in result:
        X_path += x
        Y_int += y

    return X_path, Y_int, all_class

def auto_split_data_multiprocessing(route, fold, valid_fold=None, test_fold=None, stratified=False, seed=42):
    """
    input:
        route to the directory that its subfolder are classes
        fold: how many folds
    output:
        X, Y, class corresponding
    """
    X_path, Y_int, all_class = get_data_from_phrase_multiprocessing(route)

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

def auto_split_data_multiprocessing_faster(route, valid_ratio=0.1, test_ratio=None, seed=42):
    """
    input:
        route to the directory that its subfolder are classes
        fold: how many folds
    output:
        X, Y, class corresponding
    """
    X_path, Y_int, all_class = get_data_from_phrase_multiprocessing(route)
    
    df = pd.DataFrame({'image':X_path, 'label':Y_int})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if valid_ratio is not None and test_ratio is not None:
        n_test = int(len(X_path) * test_ratio)
        n_valid = int(len(X_path) * valid_ratio)

        df_test = df[:n_test]
        df_valid = df[n_test:n_test+n_valid]
        df_train = df[n_test+n_valid:]

        X_test = df_test['image'].values
        X_valid = df_valid['image'].values
        X_train = df_train['image'].values

        Y_test = df_test['label'].values
        Y_valid = df_valid['label'].values
        Y_train = df_train['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid, X_test, Y_test

    elif valid_ratio is not None:
        n_valid = int(len(X_path) * valid_ratio)

        df_train = df[n_valid:]
        df_valid = df[:n_valid]

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

if __name__ == '__main__':
    from utils import *

    settings = get_settings()
    globals().update(settings)

    route_dataset = path_join(route, 'dataset')

    X_train, Y_train, all_class, X_valid, Y_valid = auto_split_data_multiprocessing_faster(route_dataset, valid_ratio, test_ratio, seed)
    
    print(len(all_class))
    print(len(X_train))
    print(len(X_valid))
    print(X_train[0])
    print(Y_train[0])
    