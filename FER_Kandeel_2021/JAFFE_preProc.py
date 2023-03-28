from glob import glob
import numpy as np
import PIL.Image
import shutil
import os

PATH = '.\JAFFE\jaffedbase\jaffedbase\\'

cat_dict = {'SA':'Sad',
            'SU':'Surprise',
            'HA':'Happy',
            'AN':'Angry',
            'DI':'Disgust',
            'NE':'Neutral',
            'FE':'Fear'
            }

for path, sub_dir, ims in os.walk(PATH):
    for im in ims:
        if ".tiff" in im:
            tar_loc = [fol for key, fol in cat_dict.items() if key in im][0]
            tar_path = os.path.join('.\JAFFE\JAFFE', tar_loc)
            shutil.copy2(os.path.join(path, im), tar_path)