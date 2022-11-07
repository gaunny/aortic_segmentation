import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import nibabel as nib
import re
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
# filepath = natural_sort(glob.glob(r'D:/Workspace/ct_heart2022/edge_detection/images/t/*_image.nii'))
filepath = natural_sort(glob.glob(r'D:/Workspace/ct_heart2022/test/resample_coa_89_ms1.nii.gz'))
print(len(filepath))
for f in range(len(filepath)):
    images = []
    im = nib.load(filepath[f])
    im = im.get_data()
    # float64 to uint8
    im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    im = im.astype(np.uint8)
    for i in range(im.shape[0]):
        im_slice = im[i,:,:]
        edges = cv.Canny(im_slice, 200, 250)
        images.append(edges)
    images = np.asarray(images)
    nii_image = nib.Nifti1Image(images, np.eye(4))
    name = filepath[f].split('/')[-1].split('.')[0].split('_')[1]
    print(name)
    nib.save(nii_image,r'D:/Workspace/ct_heart2022/test/resample_coa_89_ms1_edge.nii.gz'.format(name))


# plt.figure(figsize=(16,16),dpi=100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)  
# plt.imshow(edges,cmap = 'gray')
# plt.axis('off')
# cv.imwrite('sag_07_edge.png',edges)