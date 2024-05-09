# HOMEE AI 3DGS for interior environment
This repo try to improve [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) so that it can reconstruct interior environment with good quality by using ARKit data.

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate homee_3DGS
```

## data format
```shell
dataset
| ---- images
| ---- sparse
       | --- 0
             | --- cameras.txt
             | --- images.txt
             | --- point3D.txt
             | --- scene.obj (only for ARKit dataset)
```

## Running

To run the optimizer, simply use

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