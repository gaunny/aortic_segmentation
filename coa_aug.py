
#数据扩增
import re
import glob
import sitk_functions as func
import SimpleITK as sitk

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#filename_img = natural_sort(glob.glob('data/resize_data/coa_*_img_128.nii.gz'))
filename_img = natural_sort(glob.glob('data/noresize/filter_img/filter_coa_*_img.nii.gz'))
filename_label = natural_sort(glob.glob('data/noresize/label/coa_*_mask.nii.gz'))

for k in range(len(filename_img)):
    print('Load image',(k+1))
    img_sitk = sitk.ReadImage(filename_img[k])
    print(filename_img[k])
    label_sitk = sitk.ReadImage(filename_label[k])
    print(filename_label[k])
    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(img_sitk)
    min_val = filter.GetMinimum()

    # NORMAL
    sitk.WriteImage(img_sitk,'data/noresize/Augment150/normal_{}_image.nii'.format(k))
    sitk.WriteImage(label_sitk,'data/noresize/Augment150/normal_{}_label.nii'.format(k))

   #ROTATION
    [img_rot, label_rot] = func.affine_rotate(img_sitk,label_sitk,min_val)
    
    sitk.WriteImage(img_rot,'data/noresize/Augment150/rot_{}_image.nii'.format(k))
    sitk.WriteImage(label_rot,'data/noresize/Augment150/rot_{}_label.nii'.format(k))

   # SHEAR
    [img_sh,label_sh] = func.affine_shear(img_sitk,label_sitk,min_val)
    sitk.WriteImage(img_sh,'data/noresize/Augment150/sh_{}_image.nii'.format(k))
    sitk.WriteImage(label_sh,'data/noresize/Augment150/sh_{}_label.nii'.format(k))
#    
   # Intensity
    l1 = func.mult_and_add_intensity_fields(img_sitk)
    sitk.WriteImage(l1,'data/noresize/Augment150/intensity_{}_image.nii'.format(k))
    sitk.WriteImage(label_sitk,'data/noresize/Augment150/intensity_{}_label.nii'.format(k))

   # B SPLINE
    numcontrolpoints = 5
    stdDeform = 15
    dim = 3
    [img_bspline,label_bspline] = func.BSplineDeform(img_sitk,label_sitk, dim, numcontrolpoints, stdDeform,min_val)
    sitk.WriteImage(img_bspline,'data/noresize/Augment150/bspline_{}_image.nii'.format(k))
    sitk.WriteImage(label_bspline,'data/noresize/Augment150/bspline_{}_label.nii'.format(k))

#Augment_data_resize里的01被换了，后面要是重新跑网络记得重新增强
    #B SPLINE
    numcontrolpoints = 10
    stdDeform = 10
    dim = 3
    [img_bspline, label_bspline] = func.BSplineDeform(img_sitk,label_sitk, dim, numcontrolpoints, stdDeform,min_val)
    sitk.WriteImage(img_bspline,'data/noresize/Augment150/bspline2_{}_image.nii'.format(k))
    sitk.WriteImage(label_bspline,'data/noresize/Augment150/bspline2_{}_label.nii'.format(k))
    print('Finished with image',(k+1))




