#将提取到的边缘图像与原图分别通过两个通道输入
from operator import index
import os
import glob
from unicodedata import name

from cv2 import mean
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from time import time
import re
import nibabel as nib
import numpy as np
from skimage import transform
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,UpSampling2D,Activation,add,multiply,Lambda,concatenate
from tensorflow.python.layers.normalization import BatchNormalization 
#动态分配显存
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True 
lr = 1e-5
nEpochs = 30
n_classes = 2
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

imgDim = 96
labelDim = 96

tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None,imgDim,imgDim,2],name='x_train')
x_contextual = tf.placeholder(tf.float32,[None,imgDim,imgDim,3],name = 'x_train_context') 
y = tf.placeholder(tf.float32,[None,labelDim,labelDim,n_classes],name='y_train')

drop_rate = tf.placeholder(tf.float32, shape=())

def create_data(filename_img, filename_label,direction):
    images = []
    for f in range(len(filename_img)):

        a = nib.load(filename_img[f])
        a = a.get_data()
        # a2 = np.clip(a,0,180)
        im = a
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
    images = images.reshape(-1,imgDim,imgDim,1)

    labels = []
    for g in range(len(filename_label)):
        b = nib.load(filename_label[g])
        b = b.get_data()
        lab = b
        lab[lab>0] = 1
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
    
    labels_onehot = np.stack((labels == 0, labels == 1), axis = 3)
    return images, labels_onehot
def create_edge_data(filename_img,direction):
    images = []
    for f in range(len(filename_img)):

        a = nib.load(filename_img[f])
        a = a.get_data()
        # a2 = np.clip(a,0,180)
        im = a
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
    images = images.reshape(-1,imgDim,imgDim,1)
    return images


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
def dice_coef_loss(y, output):
	1 - dice_coef(y, output)

# def dice_coef(y, output): #making the loss function smooth
#     smooth = 1
#     smooth_tf = tf.constant(smooth,tf.int64)
#     y_true_f = tf.contrib.layers.flatten(tf.argmax(y,axis=-1))
#     y_pred_f = tf.contrib.layers.flatten(tf.argmax(output,axis=-1))
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2 * intersection + smooth_tf) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth_tf)
def iou_coef(y,output):
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    output_f = tf.cast(output, tf.float32)
    zeros = tf.zeros_like(output_f)
    ones = tf.ones_like(output_f)
    output_f = tf.where(output_f < 0.5, zeros, ones)
    y_f = tf.cast(y, tf.float32)
    numerator = tf.reduce_sum(y_f * output_f, axis=(1,2))
    denominator = tf.reduce_sum(y_f + output_f, axis=(1,2))
    iou = tf.reduce_mean((numerator + smooth_tf) / (denominator -numerator + smooth_tf), axis=0)
    return iou

def precision_coef(y,output):
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    output_f = tf.cast(output, tf.float32)
    zeros = tf.zeros_like(output_f)
    ones = tf.ones_like(output_f)
    output_f = tf.where(output_f < 0.5, zeros, ones)
    y_f = tf.cast(y, tf.float32)
    tp = tf.reduce_sum(y_f * output_f, axis=(1,2))
    fp = tf.reduce_sum(output_f - y_f * output_f,axis=(1,2))
    precision = tf.reduce_mean((tp+smooth_tf)/(tp+fp+smooth_tf),axis=0)
    return precision

def recall_coef(y,output):
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    output_f = tf.cast(output, tf.float32)
    zeros = tf.zeros_like(output_f)
    ones = tf.ones_like(output_f)
    output_f = tf.where(output_f < 0.5, zeros, ones)
    y_f = tf.cast(y, tf.float32)
    tp = tf.reduce_sum(y_f * output_f, axis=(1,2))
    fn = tf.reduce_sum(y_f - y_f * output_f,axis=(1,2))
    recall = tf.reduce_mean((tp+smooth_tf)/(tp+fn+smooth_tf),axis=0)
    return recall

def f1_score_coef(y,output):
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    output_f = tf.cast(output, tf.float32)
    zeros = tf.zeros_like(output_f)
    ones = tf.ones_like(output_f)
    output_f = tf.where(output_f < 0.5, zeros, ones)
    y_f = tf.cast(y, tf.float32)
    precision = precision_coef(y,output)
    recall = recall_coef(y,output)
    F1 = tf.reduce_mean((2 * precision * recall) / (precision + recall),axis=0)
    return F1

def conv2d(inputs,filters,kernel,stride,pad,name):
    with tf.name_scope(name):  #定义一块命名空间
        conv = tf.layers.conv2d(inputs, filters, kernel_size = kernel, strides = [stride,stride], padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer())
        bn = tf.layers.batch_normalization(conv, training=True)
        activation=tf.nn.relu(bn)
        
        return activation
def conv2d1(inputs,filters,kernel,stride,pad,name):
    with tf.name_scope(name):  #定义一块命名空间
        conv = tf.layers.conv2d(inputs, filters, kernel_size = kernel, strides = [stride,stride], padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer())
        activation=tf.nn.relu(conv)
        
        return activation

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
def expend_as(tensor, rep,name):
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat

def inception_block(inputs,filter_a,filter_b):
    convb1 = conv2d1(inputs,filter_a,kernel=1,stride=1,pad='same',name='convb1')
    convb2 = conv2d1(convb1,filter_b,kernel=3,stride=1,pad='same',name='convb2')
    convc1 = conv2d1(inputs,filter_a,kernel=1,stride=1,pad='same',name='convc1')
    convc2 = conv2d1(convc1,filter_b,kernel=5,stride=1,pad='same',name='convc2')
    convd1 = conv2d1(inputs,filter_b,kernel=1,stride=1,pad='same',name='convd1')
    convd2 = conv2d1(convd1,filter_b,kernel=7,stride=1,pad='same',name='convd2')
    up1 = concatenate([convb2,convc2])
    conve1 = conv2d1(up1,filter_a,kernel=3,stride=1,pad='same',name='conve1')
    up2 = concatenate([convc2,convd2])
    convf1 = conv2d1(up2,filter_a,kernel=5,stride=1,pad='same',name='convf1')
    up3 = concatenate([conve1,convf1])
    up = conv2d(up3,filter_a,kernel=3,stride=1,pad='same',name='up')
    return up
  

# -------------------------- Contracting path ---------------------------------
conv1 = inception_block(x,64,64)
pool1 = max_pool(conv1, n=2, stride=2, pad='SAME')
pool1.get_shape()
drop1 = dropout(pool1,drop_rate)
drop1.get_shape()

conv2 = inception_block(drop1,128,128)
pool2 = max_pool(conv2, n=2, stride=2, pad='SAME')
pool2.get_shape()
drop2 = dropout(pool2,drop_rate)
drop2.get_shape()

conv3 = inception_block(drop2,256,256)
pool3 = max_pool(conv3, n=2, stride=2, pad='SAME')
pool3.get_shape()
drop3 = dropout(pool3, drop_rate) 
drop3.get_shape()

conv4 = inception_block(drop3,512,512)
pool4 = max_pool(conv4, n=2, stride=2, pad='SAME')
pool4.get_shape()
drop4 = dropout(pool4, drop_rate) 
drop4.get_shape()

# -------------------------- Contextual input path ----------------------------
conv1a = inception_block(x_contextual,64,64)
pool1_2 = max_pool(conv1a,n=2,stride=2,pad='SAME')
drop1_2 = dropout(pool1_2,drop_rate) 

conv2a = inception_block(drop1_2,128,128)
pool2_2 = max_pool(conv2a,n=2,stride=2,pad='SAME')
drop2_2 = dropout(pool2_2, drop_rate)

conv3a = inception_block(drop2_2,256,256)
pool3_2 = max_pool(conv3a,n=2,stride=2,pad='SAME')
drop3_2 = dropout(pool3_2, drop_rate)

conv4a = inception_block(drop3_2,512,512)
pool4_2 = max_pool(conv4a,n=2,stride=2,pad='SAME')
drop4_2 = dropout(pool4_2, drop_rate) 

# ---------------------------- Expansive path ---------------------------------
combx = concat(pool4,pool4_2,axis=3)
conv5 = inception_block(combx,1024,1024)


up6a = transpose(conv5,filters=512,kernel=2,stride=2,pad='same',name='up6a')
up6a.get_shape()

drop5 = dropout(up6a, drop_rate) 
drop5.get_shape()
up6b = concat(up6a,conv4,axis=3)
up6b.get_shape()

conv7a = conv2d(up6b,filters=512,kernel=3,stride=1,pad='same',name = 'conv7a')
conv7a.get_shape()
conv7b = conv2d(conv7a,filters=512,kernel=3,stride=1,pad='same',name = 'conv7b')
conv7b.get_shape()
up7a = transpose(conv7b,filters=256,kernel=2,stride=2,pad='same',name='up7a')
up7a.get_shape()
drop7 = dropout(up7a, drop_rate) 
drop7.get_shape()
up7b = concat(up7a,conv3,axis=3)
up7b.get_shape()

conv8a = conv2d(up7b,filters=256,kernel=3,stride=1,pad='same',name = 'conv7a')
conv8a.get_shape()
conv8b = conv2d(conv8a,filters=256,kernel=3,stride=1,pad='same',name = 'conv7b')
conv8b.get_shape()
up8a = transpose(conv8b,filters=128,kernel=2,stride=2,pad='same',name='up7a')
up8a.get_shape()
drop8 = dropout(up8a, drop_rate) 
drop8.get_shape()
up8b = concat(up8a,conv2,axis=3)
up8b.get_shape()

conv9a = conv2d(up8b,filters=128,kernel=3,stride=1,pad='same',name = 'conv7a')
conv9a.get_shape()
conv9b = conv2d(conv9a,filters=128,kernel=3,stride=1,pad='same',name = 'conv7b')
conv9b.get_shape()

up9a = transpose(conv9b,filters=64,kernel=2,stride=2,pad='same',name='up7a')
up9a.get_shape()
drop9 = dropout(up9a, drop_rate) 
drop9.get_shape()
up9b = concat(up9a,conv1,axis=3)
up9b.get_shape()
conv10a = conv2d(up9b,filters=64,kernel=3,stride=1,pad='same',name = 'conv7a')
conv10a.get_shape()
conv10b = conv2d(conv10a,filters=64,kernel=3,stride=1,pad='same',name = 'conv7b')
conv10b.get_shape()

output = tf.layers.conv2d(conv10b, n_classes, 1, (1,1),padding ='same',activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer(), name = 'output')
output.get_shape()

filelist_train = natural_sort(glob.glob('data_new/ms_xiugai/Augment/*_image.nii'))
filelist_edge_train = natural_sort(glob.glob('data_new/ms_xiugai/canny_edge/train/*.nii'))
filelist_train_label = natural_sort(glob.glob('data_new/ms_xiugai/Augment/*_label.nii'))
# x_data, y_data = create_data(filelist_train, filelist_train_label,'axial')

filelist_test = natural_sort(glob.glob('data_new/ms_xiugai/val/*_image.nii'))
filelist_edge_test = natural_sort(glob.glob('data_new/ms_xiugai/canny_edge/val/*.nii'))
filelist_test_label = natural_sort(glob.glob('data_new/ms_xiugai/val/*_label.nii'))
# x_test, y_test = create_data(filelist_test, filelist_test_label,'axial')

x_train = {}
y_train = {}
x_edge_train = {}
for i in range(len(filelist_train)):
    img, lab = create_data([filelist_train[i]],[filelist_train_label[i]],'sag')
    x_train[i] = img
    y_train[i] = lab    

for n in range(len(filelist_edge_train)):
    img_edge = create_edge_data([filelist_edge_train[n]],'sag')
    x_edge_train[n] = img_edge

x_val = {}
x_val_edge_train = {}
y_val = {}
for i in range(len(filelist_test)):
    img_val, lab_val = create_data([filelist_test[i]],[filelist_test_label[i]],'sag')
    x_val[i] = img_val
    y_val[i] = lab_val    
for n in range(len(filelist_edge_test)):
    img_val_edge = create_edge_data([filelist_edge_test[n]],'sag')
    x_val_edge_train[n] = img_val_edge


######################################################################
##                                                                  ##
##                   Defining the training                          ##
##                                                                  ##
######################################################################

# Training-steps (honestly I have no idea what it does...)
global_step = tf.Variable(0,trainable=False)

###############################################################################
##                               Loss                                        ##
###############################################################################
# Compare the output of the network (output: tensor) with the ground truth (y: tensor/placeholder)
# In this case we use sigmoid cross entropu losss with logits
# loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, output))
# loss = dice_coef_loss(y,output)
def binary_category_focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.01
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    return K.mean(focal_loss)

loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, output))
correct_prediction = tf.equal(tf.argmax(output, axis=-1), tf.argmax(y, axis=-1))

# averaging the one-hot encoded vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dice = dice_coef(y, output)
output_iou = iou_coef(y, output)
precision = precision_coef(y,output)
recall = recall_coef(y,output)
F1 = f1_score_coef(y,output)
# Create contextual output:
pred = tf.argmax(tf.nn.softmax(output[0,:,:,:]),axis=-1) # [96, 96, 1]
predict = tf.one_hot(pred,2)
# x_img = np.expand_dims(x[0,:,:,0],axis=-1)
# print('x_img.shape:',x_img.shape)
# context = tf.concat([x_img,predict],axis=-1) #图像在0通道

context = tf.concat([x[0,:,:,:1], predict], axis=-1)

opt = tf.train.AdamOptimizer(lr,beta1,beta2,epsilon)

train_adam = opt.minimize(loss, global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

######################## Start training Session ###########################
start_time = time()
val_loss, val_accuracy, val_dice ,val_iou,val_precision,val_recall,val_f1= [], [], [],[], [], [],[]
train_loss, train_accuracy, train_dice ,train_iou,train_precison,train_recall,train_f1= [], [], [],[], [], [],[]

c = np.zeros([imgDim+1,imgDim,imgDim,3])
d = np.zeros([imgDim+1,imgDim,imgDim,3])
predictions = {}
keys = range(len(filelist_train))
for i in keys:
    predictions[i] = c

predictions_val = {}
keys = range(len(filelist_test))
for i in keys:
   predictions_val[i] = d

print("len(x_train):",len(x_train))
index_volumeID = np.repeat(range(len(x_train)),imgDim) #把range(len(x_train))重复了imgDim次
index_imageID = np.tile(range(imgDim),len(x_train)) #把range(imgDim)重复了Len(x_train)次
index_comb = np.vstack((index_volumeID,index_imageID)).T #相同列数，堆叠，再取转置
print("img.shape[0]:",img.shape[0])
print("img_val.shape[0]:",img_val.shape[0])
index_shuffle = shuffle(index_comb)
# print("index_shuffle.shape:",index_shuffle.shape)

with tf.Session() as sess:
    # Initialize
    t_start = time()

    sess.run(init)    
    
    # Trainingsloop
    for epoch in range(nEpochs):
        t_epoch_start = time()
        print('========Training Epoch: ', (epoch + 1))
        iter_by_epoch = len(index_shuffle) 
        print("index_shuffle:",len(index_shuffle))
        # print("x_train[0]:",x_train[0].shape)  #96,96,96,1
        # print("x_train[0][1,:,:,:]:",x_train[0][1,:,:,:].shape)  #96,96,1        
        for i in range(iter_by_epoch):
            t_iter_start = time()
            # a = x_train[index_shuffle[i,0]]  #a.shape:96,96,96,1
            # print(a.shape)
            # b = x_train[index_shuffle[i,1],:,:,:]  #bshape:96,96,1
            # print("b",b.shape)
            # c = x_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:] #c.shape：96,96,1
            # print("c:",c.shape)
            
                # print("len(img):",len(img)) #1924
                # print("len(img_val)",len(img_val)) #576
            x_img_batch = np.expand_dims(x_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            x_edge_batch = np.expand_dims(x_edge_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            x_batch = np.concatenate((x_img_batch,x_edge_batch),axis=-1)
            # print("predictions.shape:",predictions.shape)
            x_batch_context = np.expand_dims(predictions[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            y_batch = np.expand_dims(y_train[index_shuffle[i,0]][index_shuffle[i,1],:,:,:], axis=0)
            _,_loss,_acc,_dice,_iou,_precision,_recall,_f1,pred_out = sess.run([train_adam, loss, accuracy,dice,output_iou,precision,recall,F1,context], feed_dict={x: x_batch,x_contextual: x_batch_context, y: y_batch,drop_rate:0.5})   
            # print('pred_out.shape:',pred_out.shape)
            # print('pred_out2.shape:',predictions[index_shuffle[i,0]][index_shuffle[i,1]-1,:,:,:].shape)
            predictions[index_shuffle[i,0]][index_shuffle[i,1]-1,:,:,:] = pred_out
            # 下文
            # predictions[index_shuffle[i,0]][index_shuffle[i,1]-1,:,:,:] = pred_out
                    # predictions[index_shuffle[i,0]][index_shuffle[i,1]+1,:,:,:] = pred_out
            train_loss.append(_loss)
            train_accuracy.append(_acc)
            train_dice.append(_dice)
            train_iou.append(_iou)
            train_precison.append(_precision)
            train_recall.append(_recall)
            train_f1.append(_f1)
        # print("train_dice.sum:",np.sum(train_dice))
        #    Validation-step:
            if i==np.max(range(iter_by_epoch)):
                for n in range(len(x_val)):
                    for m in range(imgDim):
                        x_img_batch_val = np.expand_dims(x_val[n][m,:,:,:], axis=0)
                        x_edge_batch_val = np.expand_dims(x_val[n][m,:,:,:], axis=0)
                        x_batch_val = np.concatenate((x_img_batch_val,x_edge_batch_val),axis=-1)
                        y_batch_val = np.expand_dims(y_val[n][m,:,:,:], axis=0)
                        x_context_val = np.expand_dims(predictions_val[n][m,:,:,:], axis=0)
                        acc_val, loss_val,dice_val,iou_val,precision_val,recall_val,f1_val,out_context = sess.run([accuracy,loss,dice,output_iou,precision,recall,F1,context], feed_dict={x: x_batch_val, x_contextual: x_context_val, y: y_batch_val,drop_rate:1})
                        predictions_val[n][m-1,:,:,:] = out_context
                        val_loss.append(loss_val)
                        val_accuracy.append(acc_val)
                        val_dice.append(dice_val)
                        val_iou.append(iou_val)  
                        val_precision.append(precision_val)
                        val_recall.append(recall_val)
                        val_f1.append(f1_val)
            # print("val_dice:",np.sum(valid_dice))
                # print("sum_val:",np.sum(dice_val))   
        print("Epoch:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(train_loss)), 'avg_acc= ', "{:.9f}".format(np.mean(train_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(train_dice)),'avg_iou= ',"{:.9f}".format(np.mean(train_iou)),'avg_precision= ',"{:.9f}".format(np.mean(train_precison)),'avg_recall= ',"{:.9f}".format(np.mean(train_recall)),'avg_f1= ',"{:.9f}".format(np.mean(train_f1)))

        print("Validation:", (epoch + 1), '  avg_loss= ', "{:.9f}".format(np.mean(val_loss)), '  avg_acc= ', "{:.9f}".format(np.mean(val_accuracy)),'avg_dice= ', "{:.9f}".format(np.mean(val_dice)),'avg_iou= ',"{:.9f}".format(np.mean(val_iou)),'avg_precision= ',"{:.9f}".format(np.mean(val_precision)),'avg_recall= ',"{:.9f}".format(np.mean(val_recall)),'avg_f1= ',"{:.9f}".format(np.mean(val_f1)))

        # t_epoch_finish = time() 
        

    t_end = time()

    saver.save(sess,"result_ms_origin/inception_unet_context2_canny2/sag/model.ckpt")
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_loss_axial',train_loss)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_acc_axial',train_accuracy)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_dice_axial',train_dice)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_iou_axial',train_iou)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_precison_axial',train_precison)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_recall_axial',train_recall)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/train_f1_axial',train_f1)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_loss_axial',val_loss)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_acc_axial',val_accuracy)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_dice_axial',val_dice)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_iou_axial',val_iou)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_precison_axial',val_precision)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_recall_axial',val_recall)
    np.save('result_ms_origin/inception_unet_context2_canny2/train_hist/sag/valid_f1_axial',val_f1)
    print('Training Done! Total time:' + str(t_end - t_start))