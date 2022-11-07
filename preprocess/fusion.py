from PIL import Image
import glob
import nibabel as nib
import numpy as np
import cv2 as cv
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)
filepath = natural_sort(glob.glob(r'D:/Workspace/ct_heart2022/test/resample_coa_89_ms1.nii.gz'))
filepath_edge = natural_sort(glob.glob(r'D:/Workspace/ct_heart2022/test/resample_coa_89_ms1_edge.nii.gz'))

def blend_two_images(filepath,filepath_edge):
    for i in range(len(filepath)):
    # for f in filepath:
        images = []
        im = nib.load(filepath[i])
        im = im.get_data()
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = im.astype(np.uint8)
    # for h in filepath_edge:
        im2 = nib.load(filepath_edge[i])
        im2 = im2.get_data()
        im2 = (im2 - np.min(im2)) / (np.max(im2) - np.min(im2)) * 255
        im2 = im2.astype(np.uint8)
        for j in range(im.shape[0]):
            im_slice = im[j,:,:]
            # print('im_slice:',im_slice)
            im_slice2 = im2[j,:,:]
            # print('im_slice2:',im_slice2) 
            img = np.maximum(im_slice,0.95*im_slice2)
            # print('img:',img)
            images.append(img)
        images = np.asarray(images)
        nii_image = nib.Nifti1Image(images, np.eye(4))
    name = filepath[i].split('/')[-1].split('.')[0].split('_')[1]
    print(name)
    save_path = r'D:/Workspace/ct_heart2022/test/resample_coa_89_ms1_edge_fuse.nii.gz'.format(name)
    nib.save(nii_image,save_path)
        
blend_two_images(filepath,filepath_edge)


# # 图片背景透明化
# def transPNG(srcImageName):
#     img = Image.open(srcImageName)
#     img = img.convert("RGBA")
#     datas = img.getdata()
#     newData = list()
#     for item in datas:
#         if item[0] > 220 and item[1] > 220 and item[2] > 220:
#             newData.append((255, 255, 255, 0))
#         else:
#             newData.append(item)
#     img.putdata(newData)
#     img.show()
#     img.save( "new_image.png")
#     return img

# # file = Image.open('sag_07.png')
# verse = 'sag_07_edge.png'

# verse = transPNG(verse)

