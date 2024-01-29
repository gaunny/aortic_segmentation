# coa_preprocess-seg
The code for "Inner Diameter Measurement Oriented Aortic Segmentation: An Edge Enhancement and Contextual Fusion Deep Learning Method" in ICBBB 2023. https://dl.acm.org/doi/abs/10.1145/3586139.3586149

The image data is saved in **data**, and the model training and prediction results are saved in **Results**.

## Environments:
- python:3.6.13 
- TensorFlow-gpu:1.13.1
- numpy:1.19.5
- keras:2.8.0
- opencv:4.5.5.62
- nibabel:3.2.1
- scipy:1.1.0

## Softwares:
- 3D Slicer, used to manually crop out the region of interest, extract the centreline, and export vessel radius information.
- ITK-Snap, used for labelling imagesUsed for labelling images.

## Pipeline：
### pre-process：
1. **python resapmel_img.py**. The original images were resampled so that the voxel spacing of the CT images were all converted to 1 mm.
2. The image obtained from resampling is manually cropped out of the roi region, cropping out useless parts such as bones, and cropping into a [96,96,96] image. This is achieved here using the software **3D Slicer**.
3. **python filter.py**. Pre-segmentation processing is performed on the cropped image, using the MeanShift filter encapsulated in opencv, and changing the filtering parameters to achieve region smoothing and edge enhancement of the original low-quality image. And use **ITK-Snap** for manual labelling.
3. **python canny.py**. Edge extraction is performed on the filtered image using Canny operator.
4. **python fusion.py**. The edge image is fused with the filtered image.
5. **python coa_aug.py**. Enhancement of the processed image including rotation, cropping, grey scale enhancement, spline interpolation.
6. **python train_test_split.py**. Split training set and test set.

### train segmentation network
1. **python unet_axial.py,python unet_cor.py,python unet_sag.py**. Segmentation network training using unet network from each of the three orientations.
2. **python unet_*_drop.py**. Modify unet network to include drop-out for comparison.
3. Improvements: **python inception_context_canny.py**. Network with the addition of a context module and a multiscale convolution module.

### predict & fuse
1. **python predict_axial.py,python predict_cor.py,python predict_sag.py**. Prediction of unlabelled images. Note: The unlabelled image here has the same size as the network input image, uniformly $256 \times 256\times 256$, and has been pre-segmented.
2. **python fusion.py**,The predicted prob_maps of the three orientations are fused, here using the fusion strategy of **taking the average**.

### post-process
1. The predicted masks were post-processed, here using the **Active Counter (Snake)** tool in **ITK-Snap**, i.e., semi-automatic segmentation of the active contour model, where small unused regions were rounded off, leaving the aortic segmentation masks.
2. Masks were imported into **3D Slicer**, centreline extraction was performed using **VMTK**, and values for vessel internal diameters were obtained.
