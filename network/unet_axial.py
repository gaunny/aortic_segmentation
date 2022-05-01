from binascii import a2b_qp
import os
import glob
from unicodedata import name
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import re
import nibabel as nib
import numpy as np
from skimage.transform import resize
from sklearn.utils import shuffle
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
lr = 1e-5
nEpochs = 100
n_classes = 2
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

imgDim = 128
labelDim = 128

tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None,imgDim,imgDim,1],name='x_train')
#x_contextual = tf.placeholder(tf.float32,[None,imgDim,imgDim,1],name='x_train_context')
y = tf.placeholder(tf.float32,[None,labelDim,labelDim,n_classes],name='y_train')

drop_rate = tf.placeholder(tf.float32, shape=())

def create_data(filename_img, filename_label, direction):
    images = []
    for f in range(len(filename_img)):
        a = nib.load(filename_img[f])
        a = a.get_data()
        #a2 = np.clip(a,-1000,1000)
        a2 = np.clip(a,0,180)
    # a3 = np.interp(a2, (a2.min(), a2.max()), (-1, +1))
    # img = np.zeros([128,128,128]) + np.min(a3)
    # index1 = int(np.ceil((128-a.shape[2])/2))
    # index2 = int(128-np.floor((128-a.shape[2])/2))
    # a3 = img[:,:,index1:index2]
        axial_length = a2.shape[2]
        pad_length = 128 - axial_length
        print("pad_length:",pad_length)
        im = np.pad(a2,((0,0),(0,0),(0,pad_length)),'constant')
        
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
    images = images.reshape(-1, imgDim,imgDim,1)

    labels = []
    for g in range(len(filename_label)):
        b = nib.load(filename_label[g])
        b = b.get_data()
    # img = np.zeros([b.shape[0],b.shape[0],b.shape[0]])
    # index1 = int(np.ceil((img.shape[2]-b.shape[2])/2))
    # index2 = int(img.shape[2]-np.floor((img.shape[2]-b.shape[2])/2))
    # b = img[:,:,index1:index2]
        axial_length = b.shape[2]
        pad_length = 128 - axial_length
        print("pad_length:",pad_length)
        lab = np.pad(b,((0,0),(0,0),(0,pad_length)),'constant')
        # lab = resize(b,(imgDim,imgDim,imgDim),order=0)
        lab[lab>0] = 820
        if direction == 'sag':
            for i in range(lab.shape[0]):
                labels.append((lab[i,:,:]))
        if direction == 'cor':
            for i in range(lab.shape[1]):
                labels.append((lab[:,i,:]))
        if direction == 'axial':
            for i in range(lab.shape[2]):
                labels.append((lab[:,:,i]))    
    labels = np.asarray(labels)
    labels_onehot = np.stack((labels == 0, labels == 820), axis = 3)
    return images, labels_onehot



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()  #isdigt()方法字符串是否全为数字，若全是数字，为True，否则为Fasle.Python lower() 方法转换字符串中所有大写字符为小写
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def dice_coef(y, output):
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    output_f = tf.cast(output, tf.float32) 
    zeros = tf.zeros_like(output_f)
    ones = tf.ones_like(output_f)
    output_f = tf.where(output_f < 0.5, zeros, ones)  #将两张图中大于0.5的设置为1，小于0.5的设置为0  output_f维度[B,128,128,2]
    print('output_fshape:', output_f.shape)  
    #y_f维度[B,128,128,2]
    y_f = tf.cast(y, tf.float32)
    numerator = tf.reduce_sum(y_f * output_f, axis=(1,2)) #tf.reduce_sum：张量沿着指定的维度求和-->降维
    print('numeratorshape:', numerator.shape) #? 2
    denominator = tf.reduce_sum(y_f + output_f ,axis=(1,2))
    print('denominatorshape:', denominator.shape) #? 2
    gen_dice = tf.reduce_mean((2. * numerator + smooth_tf) / (denominator + smooth_tf),axis=0) #tf.reduce_mean:张量沿着指定的维度取平均-->降维
    print('gen_diceshape:', gen_dice.shape) #2
    return gen_dice

def conv2d(inputs,filters,kernel,stride,pad,name):
    with tf.name_scope(name):  #定义一块命名空间
        conv = tf.layers.conv2d(inputs, filters, kernel_size = kernel, strides = [stride,stride], padding=pad,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        return conv 

def max_pool(inputs,n,stride,pad):
    maxpool = tf.nn.max_pool(inputs, ksize=[1,n,n,1], strides=[1,stride,stride,1], padding=pad)
    return maxpool

def dropout(input1,drop_rate):
    input_shape = input1.get_shape().as_list()
    noise_shape = tf.constant(value=[1, 1, 1, input_shape[3]])
    drop = tf.nn.dropout(input1, keep_prob=drop_rate, noise_shape=noise_shape)
    return drop

def crop2d(inputs,dim):
    crop = tf.image.resize_image_with_crop_or_pad(inputs,dim,dim)
    return crop

def concat(input1,input2,axis):
    combined = tf.concat([input1,input2],axis)
    return combined

def transpose(inputs,filters, kernel, stride, pad, name):
    with tf.name_scope(name):
        trans = tf.layers.conv2d_transpose(inputs,filters, kernel_size=[kernel,kernel],strides=[stride,stride],padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer())
        return trans

conv1a = conv2d(x, filters=64, kernel=3, stride=1, pad='same', name='conv1a')
conv1a.get_shape()
conv1b = conv2d(conv1a, filters=64, kernel=3, stride=1, pad='same', name='conv1b')
conv1b.get_shape()
pool1 = max_pool(conv1b, n=2, stride=2, pad='SAME')
pool1.get_shape()

conv2a = conv2d(pool1, filters=128, kernel=3, stride=1, pad='same', name='conv2a')
conv2a.get_shape()
conv2b = conv2d(conv2a, filters=128, kernel=3, stride=1, pad='same', name='conv2b')
conv2b.get_shape()
drop2 = dropout(conv2b,drop_rate)
drop2.get_shape()
pool2 = max_pool(drop2, n=2, stride=2, pad='SAME')
pool2.get_shape()

conv3a = conv2d(pool2, filters=256, kernel=3, stride=1, pad='same', name='conv3a')
conv3a.get_shape()
conv3b = conv2d(conv3a, filters=256, kernel=3, stride=1, pad='same', name='conv3b')
conv3b.get_shape()
drop3 = dropout(conv3b,drop_rate)
drop3.get_shape()
pool3 = max_pool(drop3, n=2, stride=2, pad='SAME')
pool3.get_shape()

conv4a = conv2d(pool3, filters=512, kernel=3, stride=1, pad='same', name='conv4a')
conv4a.get_shape()
conv4b = conv2d(conv4a, filters=512, kernel=3, stride=1, pad='same', name='conv4b')
conv4b.get_shape()
drop4 = dropout(conv4b,drop_rate)
drop4.get_shape()
pool4 = max_pool(drop4, n=2, stride=2, pad='SAME')
pool4.get_shape()

# conv1a_2 = conv2d(x_contextual,filters=64,kernel=3,stride=1,pad='same',name='conv1a2')
# conv1b_2 = conv2d(conv1a_2,filters=64,kernel=3,stride=1,pad='same',name='conv1b2')
# pool1_2 = max_pool(conv1b_2,n=2,stride=2,pad='SAME')

# conv2a_2 = conv2d(pool1_2,filters=128,kernel=3,stride=1,pad='same',name = 'conv2a2')
# conv2b_2 = conv2d(conv2a_2,filters=128,kernel=3,stride=1,pad='same',name = 'conv2b2')
# drop2_2 = dropout(conv2b_2, drop_rate) 
# pool2_2 = max_pool(drop2_2,n=2,stride=2,pad='SAME')

# conv3a_2 = conv2d(pool2_2,filters=256,kernel=3,stride=1,pad='same',name = 'conv3a2')
# conv3b_2 = conv2d(conv3a_2,filters=256,kernel=3,stride=1,pad='same',name = 'conv3b2')
# drop3_2 = dropout(conv3b_2, drop_rate)  
# pool3_2 = max_pool(drop3_2,n=2,stride=2,pad='SAME')

# conv4a_2 = conv2d(pool3_2,filters=512,kernel=3,stride=1,pad='same',name = 'conv4a2')
# conv4b_2 = conv2d(conv4a_2,filters=512,kernel=3,stride=1,pad='same',name = 'conv4b2')
# drop4_2 = dropout(conv4b_2, drop_rate) 
# pool4_2 = max_pool(drop4_2,n=2,stride=2,pad='SAME')

# combx = concat(pool4,pool4_2,axis=3)

conv5a = conv2d(pool4, filters=1024, kernel=3, stride=1, pad='same', name='conv5a')
conv5a.get_shape()
conv5b = conv2d(conv5a, filters=1024, kernel=3, stride=1, pad='same', name='conv5b')
conv5b.get_shape()
drop5 = dropout(conv5b, drop_rate) 
drop5.get_shape()
up6a = transpose(drop5, filters=512, kernel=2, stride=2, pad='same', name='up6a')
up6a.get_shape()
up6b = concat(up6a, conv4b, axis=3)
up6b.get_shape()

conv7a = conv2d(up6b, filters=512, kernel=3, stride=1, pad='same', name='conv7a')
conv7a.get_shape()
conv7b = conv2d(conv7a, filters=512, kernel=3, stride=1, pad='same', name='conv7b')
conv7b.get_shape()
drop7 = dropout(conv7b, drop_rate) 
drop7.get_shape()
up7a = transpose(drop7, filters=256, kernel=2, stride=2, pad='same', name='up7a')
up7a.get_shape()
up7b = concat(up7a, conv3b, axis=3)
up7b.get_shape()

conv8a = conv2d(up7b, filters=256, kernel=3, stride=1, pad='same', name='conv8a')
conv8a.get_shape()
conv8b = conv2d(conv8a, filters=256, kernel=3, stride=1, pad='same', name='conv8b')
conv8b.get_shape()
drop8 = dropout(conv8b, drop_rate) 
drop8.get_shape()
up8a = transpose(drop8, filters=128, kernel=2, stride=2, pad='same', name='up8a')
up8a.get_shape()
up8b = concat(up8a, conv2b, axis=3)
up8b.get_shape()

conv9a = conv2d(up8b, filters=128, kernel=3, stride=1, pad='same', name='conv9a')
conv9a.get_shape()
conv9b = conv2d(conv9a, filters=128, kernel=3, stride=1, pad='same', name='conv9b')
conv9b.get_shape()
drop9 = tf.nn.dropout(conv9b, keep_prob=drop_rate) 
drop9.get_shape()
up9a = transpose(drop9, filters=64, kernel=2, stride=2, pad='same', name='up9a')
up9a.get_shape()
up9b = concat(up9a, conv1b, axis=3)
up9b.get_shape()

conv10a = conv2d(up9b, filters=64, kernel=3, stride=1, pad='same', name='conv10a')
conv10a.get_shape()
conv10b = conv2d(conv10a, filters=64, kernel=3, stride=1, pad='same', name='conv10b')
conv10b.get_shape()

output = tf.layers.conv2d(conv10b, n_classes, kernel_size=1, strides=(1,1), padding='same', activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='output')
output.get_shape()


filelist_train = natural_sort(glob.glob('data/Augment_data/*_image.nii'))
filelist_train_label = natural_sort(glob.glob('data/Augment_data/*_label.nii'))
x_data, y_data = create_data(filelist_train, filelist_train_label, 'axial')

filelist_test = natural_sort(glob.glob('data/Validation/*_image.nii'))
filelist_test_label = natural_sort(glob.glob('data/Validation/*_label.nii'))
x_test, y_test = create_data(filelist_test, filelist_test_label, 'axial')

 


global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, output))
correct_prediction = tf.equal(tf.argmax(output, axis=-1), tf.argmax(y, axis=-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dice = dice_coef(y, output)

opt = tf.train.AdamOptimizer(lr,beta1,beta2,epsilon)
train_adam = opt.minimize(loss, global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
t_start = time.time()
test_loss, test_accuracy, test_dice = [], [], []
train_loss, train_accuracy, train_dice = [], [], []

index_train = shuffle(range(x_data.shape[0]))

index_test = shuffle(range(x_test.shape[0]))

# c = np.zeros([imgDim+1,imgDim,imgDim,3])
# predictions = {}
# keys = range(len(filelist_train))
# for i in keys:
#     predictions[i] = c

# predictions_test = {}
# keys = range(len(filelist_test))
# for i in keys:
#    predictions_test[i] = c

# index_volumeID = np.repeat(range(len(x_data)),imgDim) #将数组重复imgDim次
# index_imageID = np.tile(range(imgDim),len(x_data)) #把数组复制len(x_train)次
# index_comb = np.vstack((index_volumeID,index_imageID)).T #垂直(行)按顺序堆叠数组
# index_train_shuffle = shuffle(index_comb)

#c = np.zeros([imgDim,imgDim,imgDim,3])

# predictions = {}
# keys = range(len(filelist_train))
# for i in keys:
#     predictions[i] = c

# predictions_val = {}
# keys = range(len(filelist_test))
# for i in keys:
#    predictions_val[i] = c

with tf.Session() as sess:
    t_start = time.time()
    sess.run(init)    

    for epoch in range(nEpochs):
        t_epoch_start = time.time()
        print('========Training Epoch: ', (epoch + 1))
        iter_by_epoch = len(index_train)
        index_train_shuffle = shuffle(index_train)
        for i in range(iter_by_epoch):
            t_iter_start = time.time()
            x_batch = np.expand_dims(x_data[index_train_shuffle[i],:,:,:], axis=0)
            #x_batch_context = np.expand_dims(predictions[index_train_shuffle[i],:,:,:], axis=0)
            y_batch = np.expand_dims(y_data[index_train_shuffle[i],:,:,:], axis=0)
            _,_loss,_acc,_dice= sess.run([train_adam, loss, accuracy, dice], feed_dict = {x: x_batch, y: y_batch,drop_rate: 0.5})               
            # pred_out = predictions[index_train_shuffle[i,0]][index_train_shuffle[i,1]+1,:,:,:]
            train_loss.append(_loss)
            train_accuracy.append(_acc)
            train_dice.append(_dice)

            if i==np.max(range(iter_by_epoch)):
                test_range = x_test.shape[0]
                for m in range(test_range):
                    x_batch_test = np.expand_dims(x_test[m,:,:,:], axis=0)
                    y_batch_test = np.expand_dims(y_test[m,:,:,:], axis=0)
                    #x_context_val = np.expand_dims(predictions_val[m,:,:,:], axis=0)
                    _loss_test,_acc_test,_dice_test, = sess.run([loss,accuracy,dice], feed_dict= {x: x_batch_test,y: y_batch_test,drop_rate: 1.0})
                        # pred_out = predictions_test[n][m+1,:,:,:]
                    test_loss.append(_loss_test)
                    test_accuracy.append(_acc_test)
                    test_dice.append(_dice_test)

        t_epoch_finish = time.time() 
        print("Epoch:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(train_loss)), 'avg_acc= ', "{:.9f}".format(np.mean(train_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(train_dice)),' time_epoch=', str(t_epoch_finish-t_epoch_start))
        print("Test:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(test_loss)), '  avg_acc= ', "{:.9f}".format(np.mean(test_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(test_dice)))

    t_end = time.time()
    
    saver.save(sess,"Results/unet_axial_21_aug_105_filter_pad/model.ckpt")
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/train_loss_axial',train_loss)
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/train_acc_axial',train_accuracy)
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/valid_loss_axial',test_loss)
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/valid_acc_axial',test_accuracy)
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/train_dice_axial',train_dice)
    np.save('Results/train_hist/unet_21_aug_105_filter_pad/valid_dice_axial',test_dice)
    print('Training Done! Total time:' + str(t_end - t_start))