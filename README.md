# HOMEE AI 3DGS for interior environment
This repo try to improve [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) so that it can reconstruct interior environment with good quality by using ARKit data.

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:homee-ai/gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/homee-ai/gaussian-splatting.git --recursive
```

## Setup
Our default, provided install method is based on Conda package and environment management:
```shell
cd gaussian-splatting/
conda env create --file environment.yml
conda activate homee_3DGS

# install hloc
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

## data format
### preprocess
Our preprocessor expects the following dataset structure in the source path location:
```shell
dataset
| ---- colmap
      | ---- distort_images
      | ---- sparse
            | --- 0
                  | --- calibration.json
                  | --- distort_cameras.txt
                  | --- images.txt
                  | --- images_post.txt
      | --- scene.obj (only for ARKit dataset)
```

Run the preprocess script
```shell
bash prepare_dataset.sh <path to desire dataset colmap folder> [<method1> <method2> ...]
```
After preprocessing, you will get the following dataset structure in the source path location:
```shell
dataset
| ---- colmap
      | ---- distort_images
      | ---- sparse
      | ---- post
            | ---- images
            | ---- sparse
                  | ---- offline (if at least one pose-refining method is selected)
                        | --- method1
                              | --- final
                                    | --- cameras.txt(bin)
                                    | --- images.txt(bin)
                                    | --- point3D.txt(bin)
                              | --- ...
                        | --- method2
                              | --- final
                                    | --- cameras.txt(bin)
                                    | --- images.txt(bin)
                                    | --- point3D.txt(bin)
                              | --- ...
                        | --- ...
                  | ---- online
                        | --- cameras.txt
                        | --- images.txt
                        | --- point3D.txt
                  | ---- online_loop
                        | --- cameras.txt
                        | --- images.txt
                        | --- point3D.txt
      | --- scene.obj (only for ARKit dataset)
| ---- <root>_nerfstudio (copy to directly train in nerfstudio)
      | --- images
      | --- colmap
            | ---- arkit
                  | ---- 0
                        | --- cameras.txt
                        | --- images.txt
                        | --- point3D.txt
            | ---- method1 (optional)
                  | ---- 0
                        | --- cameras.txt
                        | --- images.txt
                        | --- point3D.txt
            | ---- method2 (optional)
            | ---- ...
      
```
To run the optimizer, simply use
```shell
python train.py -s <path to dataset/colmap/post> -t <folder name under post/sparse> -m <path to desire output folder> --gs_type <gs or gs_mesh> --appearance_modeling
```
###
## Quick test
1. Download testing data from [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
2. Copy testing data to 3DGS repo. 
3. Unzip testing data. 
```shell
unzip data/tandt_db.zip 
```
4. Train
```shell
python train.py -s data/db/playroom/ -t 0 -m output/playroom --gs_type gs
```

## View result
The output folder expect to be like the following dataset structure in the model path location:
```shell
output
| ---- point_cloud 
      | --- iteration_x
            | --- point_cloud.ply
            | --- ...
      | --- ...
| ---- ...
```
Download the trained scene (.ply) to local host and visulaized in [polycam](https://poly.cam/tools/gaussian-splatting)

## Run on ARKit data
1. Download Homee ARKit data from [here](https://drive.google.com/drive/folders/12gE21_EQU7myRU-fAIZCSNQZk7geGH9I?usp=drive_link).

2. To run the optimizer, simply use
```shell
python train.py -s <path to dataset> -t <folder name under sparse> -m <path to desire output folder> --gs_type <gs or gs_mesh> --appearance_modeling
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --gs_type
  We have two gs type, gs and gs_mesh, use gs_mesh if we have scene.obj file.
  #### --appearance_modeling
  Enable appearance modeling or not.
</details>
<br>
