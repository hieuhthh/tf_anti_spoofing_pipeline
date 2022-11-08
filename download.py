import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

url = "https://drive.google.com/file/d/1z-vHZtAAyZ_WmZfVsa7LBHUwCnW6Fx0E/view?usp=share_link"
output = f"{des}/train.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1yIAjyr-kyurS3qD7YuuobhoixoZI9QBv/view?usp=share_link"
output = f"{des}/public_test.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)