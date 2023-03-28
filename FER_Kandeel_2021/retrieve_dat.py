from glob import glob
import numpy as np
import PIL.Image
import os

class dataset:
    def __init__(self, name):
        if name == "CKPLUS":
            self.path = ".\CKPLUS\CK+48\\"
        elif name == "FER2013":
            self.path = ".\FER2013\FER_2013\\"
        elif name == "JAFFE":
            self.path = ".\JAFFE\JAFFE\\"

        self.labels = os.listdir(self.path)
        self.label_size = {}
        self.dat_path = {}
        for label in self.labels:
            self.dat_path[label] = glob(self.path+label+"\*")
            self.label_size[label] = len(os.listdir(self.path+label+"\\"))

    def image_rotation(self, labels=None, angles=None, subfix_len=4):
        if labels is None:
            labels = list(self.label_size.keys())
        if angles is None:
            ROT_ANG = np.arange(-0.5, 0.5, 0.1, dtype=np.float16).round(1)
            ROT_ANG = ROT_ANG[ROT_ANG!=0]
        else:
            ROT_ANG = angles

        for tar_label in labels:
            for im_path in self.dat_path[tar_label]:
                im = PIL.Image.open(im_path)
                for angle in ROT_ANG:
                    tar_path = im_path[:(-1)*subfix_len]+'_rot'+str(angle).replace('.', '')+im_path[(-1)*subfix_len:]
                    im_rot = im.rotate(angle)
                    im_rot.save(tar_path)

    def delete_augmented_figs(self):
        for path, sub_dir, ims in os.walk(self.path):
            for im in ims:
                if "_rot" in im:
                    os.remove(os.path.join(path, im))
#                    print(os.path.join(path, im))
        return 0

    def crop(self, labels=None, subfix_len=5):
        if labels is None:
            labels = list(self.label_size.keys())

        for tar_label in labels:
            for im_path in self.dat_path[tar_label]:
                im = PIL.Image.open(im_path)
                if im.size != (48, 48):
                    #ori_width, ori_height = im.size
                    #crop_width = round((ori_width-48)*0.5, 0)
                    #crop_height = round((ori_height-48)*0.5, 0)
                    left = 65
                    right = 185
                    top = 90
                    bottom = 230
                    im_crop = im.crop((left, top, right, bottom))
                    im_new = im_crop.resize((48, 48))
                    im.close()
                    tar_path = im_path[:(-1) * subfix_len] + im_path[(-1) * subfix_len:]
                    out = im_new.convert("RGB")
                    out.save(tar_path[:-5]+".jpeg", "JPEG", quality=100)
                    os.remove(im_path)

        return 0



JAFFE_dat = dataset("JAFFE")
JAFFE_dat.crop()
#rot_angle = np.arange(0.1, 5, 0.1, dtype=np.float16).round(1)
#rot_angle = rot_angle[rot_angle!=0]
#JAFFE_dat.image_rotation(angles=rot_angle, subfix_len=5)



#CK_dat = dataset("CKPLUS")
#CK_dat.delete_augmented_figs()
#rot_angle = np.arange(0.5, 5, 0.5, dtype=np.float16).round(1)
#rot_angle = rot_angle[rot_angle!=0]
#CK_dat.image_rotation(angles=rot_angle)

#print(CK_dat.dat_path)
#input()
#FER_dat = dataset("FER2013")
#FER_dat.delete_augmented_figs()
#rot_angle = np.arange(0.5, 5, 0.5, dtype=np.float16).round(1)
#rot_angle = rot_angle[rot_angle!=0]
#FER_dat.image_rotation(labels=['disgust'], angles=rot_angle)
#print(FER_dat.label_size)