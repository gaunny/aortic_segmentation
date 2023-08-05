import SimpleITK as sitk
import numpy as np
import glob 
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def transform(image,newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    # 设置一个Filter
    resample = sitk.ResampleImageFilter()
    # 原来的体素块尺寸
    originSize = image.GetSize()
    # 原来的体素间的距离
    originSpacing = image.GetSpacing()
    #print(originSpacing)
    # newSize = np.array(newSize, float)
    # newSpacing = np.array(newSpacing, float)
    # factor = originSpacing / newSpacing
    # newSize = originSize * factor
    # newSpacing = newSpacing.astype(np.int)
    newSize = [
        int(np.round(originSize[0] * originSpacing[0] / newSpacing[0])),
        int(np.round(originSize[1] * originSpacing[1] / newSpacing[1])),
        int(np.round(originSize[2] * originSpacing[2] / newSpacing[2]))
    ]

    # 默认像素值（2）
    # resample.SetDefaultPixelValue(image.GetPi);
    # 沿着x,y,z,的spacing（3）
    # The sampling grid of the output space is specified with the spacing along each dimension and the origin.
    resample.SetOutputSpacing(newSpacing)
    # 设置original
    resample.SetOutputOrigin(image.GetOrigin())
    # 设置方向
    resample.SetOutputDirection(image.GetDirection())
    resample.SetSize(newSize)
    # 设置插值方式
    resample.SetInterpolator(resamplemethod)
    # 设置transform
    resample.SetTransform(sitk.Euler3DTransform())
    # 默认像素值   resample.SetDefaultPixelValue(image.GetPixelIDValue())
    return resample.Execute(image)

if __name__ == '__main__':
    file_img = natural_sort(glob.glob('data/noresize/img/coa_*_img.nii.gz'))
    # file_label = natural_sort(glob.glob(r'D:/celiang/images_good/filter/coa_09_mask.nii.gz'))
    for i in range(len(file_img)):
        print("Start_", i, '_converse')

        image = sitk.ReadImage(file_img[i])  # 读取image文件
        # print(image.GetOrigin())
        # print(image.GetSpacing())
        # print(image.GetSize())
        # label=sitk.ReadImage(file_label[i]) #读取mask

        newImage=transform(image,[1,1,1],sitk.sitkLinear)  #这里[1,1,1]可以改成任意spacing值
        # newLabel=transform(label,[1,1,1])# 不添加插值方式时，默认为最近邻插值
        # mask请用最近邻插值，image用线性插值
        name = file_img[i].split('/')[-1].split('.')[0]
        name_data = 'data/noresize/resample_img/{}.nii.gz'.format(name)
        # name_label='coa_09_mask.nii.gz' # label存储path

        sitk.WriteImage(newImage, name_data)
        # sitk.WriteImage(newLabel, name_label)
