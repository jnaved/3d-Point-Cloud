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
&#8594; install the requirements mentioned in the requirements.txt  
You can do so by   
```
pip install -r requirements.txt
```
If it doesnt work use  
```
python -m pip install -r requirements.txt
```
### 2. Data Preparation
Download the 3D KITTI detection dataset from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip).  
You need to Login First to do so. Once logged in, download the velodyne data from object->bird_view section  
The downloaded data includes:
- Velodyne point clouds( 29 GB)
- Training labels of objects data set(5 MB)
- Camera calibration matrices of object data set(16 MB)
- Left color images of object data set(12 GB)
Construct them according to the folder structure below
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