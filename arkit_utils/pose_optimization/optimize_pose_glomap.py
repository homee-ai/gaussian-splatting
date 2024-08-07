import pycolmap
import argparse
from hloc.triangulation import create_db_from_model
from hloc.utils.read_write_model import Camera, Image, Point3D, CAMERA_MODEL_NAMES
from hloc.utils.read_write_model import write_model, read_model
from hloc import extract_features, match_features, pairs_from_poses, triangulation
from pathlib import Path
import logging
import os
import numpy as np

def create_db(input_dir, output_dir):
    assert input_dir.exists(), input_dir
    assert output_dir.exists(), output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "database_compare.db"
    reference = pycolmap.Reconstruction(input_dir)

    create_db_from_model(reference, database)


def prepare_pose_and_intrinsic_prior(dataset_base) :
    dataset_dir = Path(dataset_base)
    
    # step1. Write ARKit pose (in COLMAP ccordinate) to COLMAP images
    images = {}
    with open(dataset_base + "/post/sparse/online_loop/images.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])

                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    
    # step2. Write ARKit undistorted intrinsic to COLMAP cameras
    cameras = {}
    with open(dataset_base +  "/post/sparse/online_loop/cameras.txt", "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)


    # Empty 3D points.
    points3D = {}

    print('Writing the COLMAP model...')
    colmap_arkit_base = dataset_dir / 'post' / 'sparse' /'offline'
    colmap_arkit =  colmap_arkit_base / 'raw'
    colmap_arkit.mkdir(exist_ok=True, parents=True)
    write_model(images=images, cameras=cameras, points3D=points3D, path=str(colmap_arkit), ext='.bin')

    return colmap_arkit



def optimize_pose_by_glomap(dataset_base, n_matched = 10):
    dataset_dir = Path(dataset_base)
    colmap_arkit_base = dataset_dir / 'post' / 'sparse' /'offline'
    colmap_arkit =  colmap_arkit_base / 'raw'
    outputs = colmap_arkit_base / 'hloc'
    outputs.mkdir(exist_ok=True, parents=True)

    images = dataset_dir / 'post' / 'images'
    sfm_pairs = outputs / 'pairs-sfm.txt'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    references = [str(p.relative_to(images)) for p in images.iterdir()]
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superglue']

    extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    pairs_from_poses.main(colmap_arkit, sfm_pairs, n_matched)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
   
    colmap_sparse = outputs / 'colmap_sparse'
    colmap_sparse.mkdir(exist_ok=True, parents=True)

    reconstruction = triangulation.main(
        colmap_sparse,  # output model
        colmap_arkit,   # input model
        images,
        sfm_pairs,
        features,
        matches)

    glomap = outputs / 'glomap'
    glomap.mkdir(exist_ok=True, parents=True)

    glomap_cmd = f"glomap mapper \
    --database_path {colmap_sparse}/database.db \
    --image_path {images} \
    --output_path {glomap}"
    exit_code = os.system(glomap_cmd)
    if exit_code != 0:
        logging.error(f"glomap mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ARkit pose using COLMAP")
    parser.add_argument("--input_database_path", type=str, default="data/arkit_pose/study_room/arkit_undis")
    args = parser.parse_args()

    input_database_path = args.input_database_path

    prepare_pose_and_intrinsic_prior(input_database_path)
    optimize_pose_by_glomap(input_database_path)
    # run glomap