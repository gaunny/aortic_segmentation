# coa_preprocess-seg
关于coa病人心脏CT图像的处理与分割任务  

其中，图像数据保存在data中，模型训练及预测结果保存在Results中

## 运行环境：
- python：3.6.13 
- TensorFlow-gpu：1.13.1
- numpy:1.19.5
- keras:2.8.0
- opencv:4.5.5.62
- nibabel:3.2.1
- scipy:1.1.0

## 用到的软件：
- 3D Slicer，用于手动裁剪出感兴趣区域，提取中心线，导出血管半径信息
- ITK-Snap，用于标注图像

## 运行步骤
### pre-process：
1. 将原图手动裁剪出roi区域，裁掉骨骼等无用部分。这里使用软件**3D Slicer**实现。
2. **运行filter.py**。对裁剪出的图像进行预分割处理，使用opencv中封装好的MeanShift滤波，更改滤波参数，实现原低质量图像的区域平滑与边缘增强。并使用**ITK-Snap**进行手动标注。
3. **运行resize_pad.py**。对增强的图像进行尺寸统一处理，全部统一成$256 \times 256\times 256$大小的图片，其中，原图像不足部分进行补零处理。
4. **运行coa_aug.py**。对处理后的图像进行增强处理，包括旋转，裁切，灰度增强，样条插值。
5. **运行train_test_split.py**。进行训练集与测试集的分类。

### train segmentation network
1. 分别运行**unet_axial.py,unet_cor.py,unet_sag.py**，使用unet网络分别从三个面进行分割网络训练
2. 修改unet网络，加入drop-out，训练**unet_*_drop.py**进行对比

### predict & fuse
1. 分别运行**predict_axial.py,predict_cor.py,predict_sag.py**，对未标注图像进行预测。注：这里的未标注图像与网络输入图像尺寸相同，统一为$256 \times 256\times 256$，并且经过了预分割处理。
2. 运行**fusion.py**,将三个面的预测prob_maps融合，这里采用**取平均**的融合策略。

### post-process
-  对预测出的掩码进行后处理，这里采用**ITK-Snap**中的**Active Counter(Snake)**工具，即主动轮廓模型半自动分割，将不用的小区域舍去，留下主动脉的分割掩码。