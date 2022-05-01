import numpy as np
import re
import nibabel as nib
import glob
from skimage.transform import resize
from  scipy import ndimage
import tensorflow as tf
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

files_p_axial = natural_sort(glob.glob('Results/Predictions/testprobmaps/coa77/test_prob_maps_axial_*.npz'))
files_p_sag = natural_sort(glob.glob('Results/Predictions/testprobmaps/coa77/test_prob_maps_sag_*.npz'))
files_p_cor = natural_sort(glob.glob('Results/Predictions/testprobmaps/coa77/test_prob_maps_cor_*.npz'))

for n in range(len(files_p_axial)):
               
        axial_data = np.load(files_p_axial[n])
        prob_maps_axial = axial_data['prob_maps']
        sag_data = np.load(files_p_sag[n])
        prob_maps_sag = sag_data['prob_maps']
        cor_data = np.load(files_p_cor[n])
        prob_maps_cor = cor_data['prob_maps']
    
        cor = prob_maps_cor.transpose(1,0,2,3)
        axial = prob_maps_axial.transpose(1,2,0,3)
    
        axial_maps = np.zeros([128,128,128,2])
        axial_maps[axial > 0.5] = 1
        axial_maps[axial < 0.5] = 0
    
        cor_maps = np.zeros([128,128,128,2])
        cor_maps[cor > 0.5] = 1
        cor_maps[cor < 0.5] = 0
    
        sag_maps = np.zeros([128,128,128,2])
        sag_maps[prob_maps_sag > 0.5] = 1
        sag_maps[prob_maps_sag < 0.5] = 0
    
        np.savez('Results/Predictions/final/coa77/test_segments_axial_{}'.format(n),prob_maps=axial_maps)
        np.savez('Results/Predictions/final/coa77/test_segments_cor_{}'.format(n),prob_maps=cor_maps)
        np.savez('Results/Predictions/final/coa77/test_segments_sag_{}'.format(n),prob_maps=sag_maps)
    
files_segment_axial = natural_sort(glob.glob('Results/Predictions/final/coa77/test_segments_axial_*.npz'))
files_segment_cor = natural_sort(glob.glob('Results/Predictions/final/coa77/test_segments_cor_*.npz'))
files_segment_sag = natural_sort(glob.glob('Results/Predictions/final/coa77/test_segments_sag_*.npz'))

for m in range(len(files_segment_axial)):
    axial_segment = np.load(files_segment_axial[n])
    axial_seg = axial_segment['prob_maps']
    cor_segment = np.load(files_segment_cor[n])
    cor_seg = cor_segment['prob_maps']
    sag_segment = np.load(files_segment_sag[n])
    sag_seg = sag_segment['prob_maps']
    
    stack = np.stack((sag_seg,cor_seg,axial_seg),axis=-1)  #扩维拼接，变成(128,128,128,2,3)
    #fused = np.maximum.reduce(stack,-1)
    fused = np.mean(stack,-1)
    fused_maps = np.zeros([128,128,128,2])
    fused_maps[fused > 0]=1
    fused_maps[fused == 0] =0
    delete = np.delete(fused_maps,slice(-1),axis=-1)
    delete2 = np.squeeze(delete)
    new_image = nib.Nifti1Image(delete2,np.eye(4))
    nib.save(new_image,'Results/Predictions/nii/coa77/prediction_label_*.nii.gz'.format(m))


