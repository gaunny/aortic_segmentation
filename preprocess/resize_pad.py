#将要预测的图片全部resize成256,256,256,其中每个面缺少的补成背景，即补零

import nibabel as nib
import glob
from skimage import transform
import numpy as np
import re 
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

filepath = natural_sort(glob.glob('data/noresize/Validation/*.nii'))

for i in range(len(filepath)):
    coa = nib.load(filepath[i])
    img = coa.get_data()
    cor_sag_length = img.shape[0]
    axial_length = img.shape[2]
    cor_sag_pad_length = 256 - cor_sag_length
    axial_pad_length = 256 - axial_length
    print('第 %d 个pad_length:%d,%d' %(i,cor_sag_pad_length,axial_pad_length))
   
    img_pad = np.pad(img,((0,cor_sag_pad_length),(0,cor_sag_pad_length),(0,axial_pad_length)),'constant')
    affine_img = coa.affine
    new_img = nib.Nifti1Image(img_pad,affine_img)
    f_name = filepath[i]
    f_name = f_name.split('/')[-1].split('.')[0]
    print(f_name)
    nib.save(new_img,'data/256pad/validationdata/pad256_{}.nii'.format(f_name))
    # nib.save(new_img,'data/Test/testdata_resize_pad/{}_128.nii.gz'.format(f_name))
