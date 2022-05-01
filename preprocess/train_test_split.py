import random
import os
import shutil
import glob
pathDir_img = glob.glob('data/noresize/Augment150/*_image.nii')


filenumber = len(pathDir_img)
rate = 0.2
picknumber = int(filenumber*rate)
sample_img = random.sample(pathDir_img, picknumber)

print("========开始移动图片=======")
for sample in sample_img:
    sample = sample.split('/')[-1].split('.')[0].split('_image')[0]
    filename_label = 'data/noresize/Augment150/{}_label.nii'.format(sample)
    file_label = filename_label.split('/')[-1]
    file_label_tar = 'data/noresize/Validation/{}'.format(file_label)
    shutil.move(filename_label, file_label_tar )
    print(filename_label)
    filename_image = 'data/noresize/Augment150/{}_image.nii'.format(sample)
    print(filename_image)
    file_image_tar = 'data/noresize/Validation/{}_image.nii'.format(sample)
    shutil.move(filename_image,  file_image_tar)
    
print("=======移动图片完成==========")