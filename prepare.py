import multiprocessing

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'dataset')

# mkdir(des)
force_mkdir(des)

def count_file(route):
    file_count = sum(len(files) for _, _, files in os.walk(route))
    return file_count

def move_folder(route, pad=None):
    if pad is None:
        pad = route.split('/')[-1]

    print(pad, len(os.listdir(route)))

    for filename in os.listdir(route): 
        from_folder = route + '/' + filename
        to_folder = des + '/' + pad + filename

        if len(from_folder) < 1:
            print("Empty folder:", from_folder)
            continue

        try:
            shutil.copytree(from_folder, to_folder)
        except:
            print('Already have:', from_folder)

def move_folder_multiprocessing(route, pad=None):
    if pad is None:
        pad = route.split('/')[-1]
    
    print(pad, len(os.listdir(route)))

    global task_move_folder

    def task_move_folder(route, list_cls):
        for cl in list_cls:
            from_folder = os.path.join(route, cl)
            to_folder = os.path.join(des, pad + cl)

            if len(from_folder) < 1:
                print("Empty folder:", from_folder)
                continue

            try:
                shutil.copytree(from_folder, to_folder)
            except:
                print('Already have:', from_folder)

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    processes = []

    all_class = sorted(os.listdir(route))
    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    for i in range(cpu_count):
        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task_move_folder, args=(route,list_cls,))
        processes.append(p)

    result = [p.get() for p in processes]

print('N class before:', len(os.listdir(des)))
print('N image before:', count_file(des))

from_folder = '/home/huynx2/hieunmt/other/tf_img_pipeline/unzip/dataset'
# move_folder(from_folder)
move_folder_multiprocessing(from_folder)

print('N class:', len(os.listdir(des)))
print('N image:', count_file(des))

    