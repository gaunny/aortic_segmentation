import cv2
import numpy as np
import glob 
import nibabel as nib
filepath = glob.glob('D:/celiang/images_good/filter/coa_65_img.nii.gz')


for f in range(len(filepath)):
        print(len(filepath))
        images = []
        im = nib.load(filepath[f])
        im = im.get_data()
        # float64 to uint8
        im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
        im = im.astype(np.uint8)

        for i in range(im.shape[0]):
            
            im_slice = im[i,:,:]
            im_slice = cv2.cvtColor(im_slice, cv2.COLOR_GRAY2BGR)
            print(im_slice.shape)
            print(type(im_slice))
            #dst = cv2.pyrMeanShiftFiltering(src=im_slice, sp=15, sr=20)
            dst = cv2.pyrMeanShiftFiltering(src=im_slice, sp=5, sr=8)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            print(dst.shape)
            print(type(dst))
            #print(type(dst)) numpy.ndarray
            #dst = dst.reshape(im.shape[0],im.shape[1],im.shape[2])
            #list_dst = dst.tolist()
            #print(type(list_dst))
            #images = []
            images.append(dst)
        images = np.asarray(images)
            
        #images = np.asarray(images)
        print('1:',images.shape)
        nii_image = nib.Nifti1Image(images, np.eye(4))
    
        name = filepath[f].split('/')[-1].split('.')[0]
        print(name)
        save_path = 'D:/celiang/images_good/filter/filter_{}.nii.gz'.format(name)
        nib.save(nii_image,save_path)
    


