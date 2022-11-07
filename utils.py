import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
import os
import shutil
from os.path import join as path_join
import yaml

def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu_instance, True)

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Using TPU')
    except:
        strategy = tf.distribute.MirroredStrategy()
        if strategy.num_replicas_in_sync == 1:
            strategy = tf.distribute.get_strategy()
            print('Using 1 GPU')
        else:
            print('Using GPUs')
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy
    
def seedEverything(seed):
    def seedTF(seed):
        tf.random.set_seed(seed)

    def seedBasic(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    seedBasic(seed)
    seedTF(seed)

def visual_save_metric(his, metric):
    val_metric = 'val_' + metric

    print(f'BEST {val_metric}:', np.min(his.history[val_metric]), 'at epoch:', np.argmin(his.history[val_metric]) + 1)

    plt.figure()
    plt.plot(his.history[metric], label=f'train {metric}')
    plt.plot(his.history[val_metric], label=f'test {metric}')
    plt.title(f'Plot History: Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend([f'Train {metric}', f'Test {metric}'], loc='upper left')
    plt.show()
    plt.savefig(f'plot_{metric}.png')

def rmdir(route):
    try:
        shutil.rmtree(route)
    except:
        pass
    
def mkdir(route):
    try:
        os.mkdir(route)
    except:
        pass

def force_mkdir(route):
    rmdir(route)
    mkdir(route)

def get_settings(file_setting='setting.yaml'):
    with open(file_setting) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    return info_dict

if __name__ == '__main__':
    set_memory_growth()
    strategy = auto_select_accelerator()
    load_var()
    print(route)