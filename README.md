# ICCVW21-TradiCV-Survey-of-LiDAR-Cluster

## Motivation
In contrast to popular end-to-end deep learning LiDAR panoptic segmentation solutions, we propose a hybrid method with an existing semantic segmentation network to extract semantic information and a traditional LiDAR point cloud cluster algorithm to split each instance object. We argue geometry-based traditional clustering algorithms are worth being considered by showing a state-of-the-art performance among all published end-to-end deep learning solutions on the panoptic segmentation leaderboard of the SemanticKITTI dataset. To our best knowledge, we are the first to attempt the point cloud panoptic segmentation with clustering algorithms. Therefore, instead of working on new models, we give a comprehensive technical survey in this paper by implementing four typical cluster methods and report their performances on the benchmark. Those four cluster methods are the most representative ones with real-time running speed. They are implemented with C++ in this paper and then wrapped as a python function for seamless integration with the existing deep learning frameworks.

<br />
<img src="https://github.com/placeforyiming/ICCVW21-LiDAR-Panoptic-Segmentation-TradiCV-Survey-of-Point-Cloud-Cluster/blob/master/examples.png?raw=true" alt="Figure" style="width: 840px; height: 500px;" hspace="10" align="left"/>
<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />

## Dataset Organization

    ICCVW21-LiDAR-Panoptic-Segmentation-TradiCV-Survey-of-Point-Cloud-Cluster
    ├──  Dataset
    ├        ├── semanticKITTI                 
    ├            ├── semantic-kitti-api-master         
    ├            ├── semantic-kitti.yaml
    ├            ├── data_odometry_velodyne ── dataset ── sequences ── train, val, test         # each folder contains the corresponding sequence folders 00,01...
    ├            ├── data_odometry_labels ── dataset ── sequences ── train, val, test           # each folder contains the corresponding sequence folders 00,01...
    ├            └── data_odometry_calib    
    ├──  method_predictions ── sequences

## How to run

```` 
```
docker pull pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime 
```
````
Install dependency packages:
```` 
```
bash install_dependency.sh
```
````
Compile specific clusters 
```` 
```
cd PC_cluster
cd ScanLineRun_cluster/Euclidean_cluster/depth_cluster/SuperVoxel_cluster
bash prepare_packages.sh/prepare_pybind.sh
bash build.sh
```
````
Note, prepare_packages.sh may redundantly install packages as clusters are supposed to be used independently. 

One can download the predicted validation results of Cylinder3D from here:
https://drive.google.com/file/d/1QkV8zmRaOAgAZse5CGtlmijcLJVnh7XP/view?usp=sharing

We get the prediction of validation 08 sequence by using the provided checkpoint of Cylinder3D. Thanks for sharing the code!

After downloading, unzip the 08 file, put it inside ./method_predictions/sequences/

It looks like ./method_predictions/sequences/08/predictions/*.label

Run the cluster algorithm
```` 
```
python semantic_then_instance_post_inferece.py
```
````
It should keep updating the visualization figure output_example.png, and overwrite predicted labels in ./method_predictions/sequences/08/predictions/

One can unzip 08 again if wants to run the cluster algorithm again.

Some parameters can be tuned in args parser.


After generating the predicted panoptic label on validation set, one can simply run:
````
```
bash evaluation_panoptic.sh
```
````
Some changes of local path may need to be done. Just follow the error to change them, should be easy. 

The reported numbers should be exactly the same as the paper since traditional methods have no randomness. 

## Publication ##
Please cite the paper if you use this code:

```
@inproceedings{zhao2021technical,
  title={A Technical Survey and Evaluation of Traditional Point Cloud Clustering Methods for LiDAR Panoptic Segmentation},
  author={Zhao, Yiming and Zhang, Xiao and Huang, Xinming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2464--2473},
  year={2021}
}


```
