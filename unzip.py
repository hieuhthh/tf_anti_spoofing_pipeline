import zipfile

from utils import *

settings = get_settings()
globals().update(settings)

from_dir = path_join(route, 'download')
des = path_join(route, 'unzip')
mkdir(des)

filename = 'more_dataset'
zip_file = path_join(from_dir, filename + '.zip')
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(des)