# 3d-Point-Cloud-Detection on LiDAR Point Clouds
## Feature
&#9745; Accurate 3D object detection based on Lidar data  
&#9745; Release pre-trained model  

## Highlights
&#9745; The work has been referred from [Github](https://github.com/maudzung/SFA3D)  
&#9745; The introduction and explanation on this project is here 
[Youtube Link](https://www.youtube.com/watch?v=cPOtULagNnI&t=4858s)  

## Technical Details
Technical Details of the implementation are [here](https://github.com/jnaved/3d-Point-Cloud/edit/main/Technical_Details.md)

## How to
### 1. Setting Up
&#8594; Set up a virtual environment(optional)  
```
git clone https://github.com/jnaved/3d-Point-Cloud.git Point_Cloud
cd Point_Cloud/
```
&#8594; Install the requirements mentioned in the requirements.txt  
You can do so by   
```
pip install -r requirements.txt
```
If it doesnt work use  
```
python -m pip install -r requirements.txt
```
### 2. Data Preparation
Download the 3D KITTI detection dataset from links below.  
The downloaded data includes:
- [Velodyne point clouds( 29 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
- [Training labels of objects data set(5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
- [Camera calibration matrices of object data set(16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [Left color images of object data set(12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
Construct them according to the folder structure given below

### 3. Run Code
**1. Visualize the data**  

to visualize the data, execute:
```
cd major_project/data_process/
python kitti_dataset.py
```
**2. Training**  

This training is done only on single machine and single gpu. The command is as follows:
```
python train.py --gpu_idx 0
```
**3. Testing**  

To view the test results, execute the command:
```
python test.py --gpu_idx 0
```
**4. Accuaracy check**  

To calculate accuracy, execute:
```
python accuracy.py --gpu_idx 0
```
The accuracy is done based on IoU method(Intersection over Union)  

**5. View the live data**  

The dataset for the live feed detection is included in the code. Run the following command to see it.
```
python demo_2_sides.py --gpu_idx 0
```
The output generated will be video file of type .avi  
You need to download a media player than can play the .avi files.

## Folder Structure

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── logs
    ├── fpn_resnet_18
└── major_project/
    ├── config/
    │   ├── __init__.py
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── __init__.py
    │   ├── demo_dataset.py
    │   ├── kitti_bev_utils.py
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    │   └── transformation.py
    ├── losses/
    │   ├── __init__.py
    │   ├── losses.py
    ├── models/
    │   ├── __init__.py
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── __init__.py
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── lr_scheduler.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── accuracy.py
    ├── calculate_accuracy.py
    ├── loss_plot.py    
    ├── video_output.py
    ├── test.py
    └── train.py
└── plot/
    ├── __init__.py
    ├── training_loss.txt
    ├── validation_loss.txt
├── README.md 
├── Technical_Details.md
└── requirements.txt
```