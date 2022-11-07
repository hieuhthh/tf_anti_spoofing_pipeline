import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

url = "https://drive.google.com/file/d/13lEWQBOdFZk7ZDxwYW6HSHgRIUu3w_68/view?usp=sharing"
output = f"{des}/more_dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)