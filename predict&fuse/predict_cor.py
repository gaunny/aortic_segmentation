import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import glob
import re
import time
from skimage.transform import resize
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
imgDim = 128

##############################################################################
###                              Data functions                         ######
##############################################################################
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def create_image(filename_img,direction):
    images = []
    a = nib.load(filename_img)
    a = a.get_data()
    # Normalize:
    a2 = np.clip(a,0,180)
    # a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
    # Reshape:
    #img设置？
    # img = np.zeros([512,512,512])+np.min(a3)
    # index1 = int(np.ceil((512-a.shape[2])/2))
    # index2 = int(512-np.floor((512-a.shape[2])/2))
    # img[:,:,index1:index2] = a3
    im = resize(a2,(imgDim,imgDim,imgDim),order=0)
    if direction == 'sag':
        for i in range(im.shape[0]):
            images.append((im[i,:,:]))
    if direction == 'cor':
        for i in range(im.shape[1]):
            images.append((im[:,i,:]))
    if direction == 'axial':
        for i in range(im.shape[2]):
            images.append((im[:,:,i]))            
    images = np.asarray(images)
    #将切片数量放到第0维
    images = np.expand_dims(images,axis=-1)
    print(images.shape) 
    return images

# Load test data
filelist_test = natural_sort(glob.glob('data/Augment_data_resize/coa77/*_0_image.nii')) # list of file names
print("====================== LOAD COR NETWORK: ===========================")
t_start1 = time.time()
# Doing predictions with the model 
tf.reset_default_graph()      

new_saver = tf.train.import_meta_graph('Results/unet_cor_21_aug_105_filter_pad/model.ckpt.meta')

with tf.Session() as sess:
    new_saver.restore(sess, tf.train.latest_checkpoint('Results/unet_cor_21_aug_105_filter_pad/'))
    graph = tf.get_default_graph()       
    x = graph.get_tensor_by_name("x_train:0")
    op_to_restore = graph.get_tensor_by_name("output/Softmax:0") #ME

    for i in range(len(filelist_test)):
        print('Processing test image', (i+1),'out of',(np.max(range(len(filelist_test)))+1))
        # Find renderings corresponding to the given name
        prob_maps = []
        x_test = create_image(filelist_test[i],'cor')
        for k in range(x_test.shape[0]):
            #取出x_test第0维，即切片数量，为128.
            x_test_image = np.expand_dims(x_test[k,:,:,:], axis=0)#x_test_image维度为[1,128,128,1]
            #过模型。y_output的维度为[1,128,128,2]
            y_output = sess.run(tf.nn.softmax(op_to_restore), feed_dict={x: x_test_image,'Placeholder:0':1.0})
            #把第0维，即batch_size的维度丢掉
            prob_maps.append(y_output[0,:,:,:])
        #输出prob_maps维度：[128,128,128,2]
        np.savez('Results/Predictions/testprobmaps/coa77/test_prob_maps_cor_{}'.format(i),prob_maps=prob_maps)                            
    print("================ DONE WITH TEST PREDICTIONS! ==================")  

t_end1 = time.time()
print("================ DONE WITH COR PREDICTIONS! ==================")
print("time:",t_end1-t_start1)



