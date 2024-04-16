# 3D Object Detection based on 3D LiDAR Point Clouds

**Technical details of the implementation**

## 1. Network architecture

- The **ResNet-based Keypoint Feature Pyramid Network** (KFPN) that was proposed in [RTM3D paper](https://arxiv.org/pdf/2001.03343.pdf).  
- **Input**: 
    - The model takes a birds-eye-view (BEV) map as input. 
    - The BEV map is encoded by height, intensity, and density of 3D LiDAR point clouds. Assume that the size of the BEV input is `(H, W, 3)`.

- **Outputs**: 
    - Heatmap for main center with a size of `(H/S, W/S, C)` where `S=4` _(the down-sample ratio)_, and `C=3` _(the number of classes)_
    - Center offset: `(H/S, W/S, 2)`
    - The heading angle _(yaw)_: `(H/S, W/S, 2)`. The model estimates the **im**aginary and the **re**al fraction (`sin(yaw)` and `cos(yaw)` values).
    - Dimension _(h, w, l)_: `(H/S, W/S, 3)`
    - `z` coordinate: `(H/S, W/S, 1)`

- **Targets**: **7 degrees of freedom** _(7-DOF)_ of objects: `(cx, cy, cz, l, w, h, θ)`
   - `cx, cy, cz`: The center coordinates.
   - `l, w, h`: length, width, height of the bounding box.
   - `θ`: The heading angle in radians of the bounding box.
   
- **Objects**: Cars, Pedestrians, Cyclists.

## 2. Losses function

- For main center heatmap: Used `focal loss`

- For heading angle _(yaw)_: The `im` and `re` fractions are directly regressed by using `l1_loss`

- For `z coordinate` and `3 dimensions` (height, width, length), I used `balanced l1 loss` that was proposed by the paper
 [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/pdf/1904.02701.pdf)

## 3. Training in details

- Set uniform weights to the above components of losses. (`=1.0` for all)
- Number of epochs: 200.
- Learning rate scheduler: `cosine`, initial learning rate: 0.001.
- Batch size: `8` (on a single GTX 1650Ti).

## 4. Inference

- A `3 × 3` max-pooling operation was applied on the center heat map, then only `50` predictions whose 
center confidences are larger than 0.2 were kept.
- The heading angle _(yaw)_ = `arctan`(_imaginary fraction_ / _real fraction_)

## 5. Result

- A .avi file format will be geneated for video input. Example case [here](https://github.com/jnaved/3d-Point-Cloud/blob/main/result/fpn_resnet_18/2011_09_26_drive_0014_sync_both_2_sides.avi)

## 6. Evaluation Technique

- Intersection over Union(IoU) method has been used here. Obtained a value of 0.87 with 200 epochs.

## 7. How to expand the work

- The model could be trained with more classes and with a larger detected area by modifying configurations in
the [config/kitti_dataset.py](https://github.com/jnaved/3d-Point-Cloud/blob/main/major_project/config/kitti_config.py) file. This could lead to better results.