import os
import shutil
import argparse
from pathlib import Path

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_directory(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        print(f"Warning: Source directory {src} does not exist. Skipping this copy operation.")

def main(args):
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"

    # Create new directory structure
    create_directory(output_root)
    
    colmap_dir = output_root / "colmap"
    create_directory(colmap_dir)
    
    # Copy arkit data
    arkit_src = root_path / "post" / "sparse" / "online_loop"
    arkit_dst = colmap_dir / "arkit" / "0"
    copy_directory(arkit_src, arkit_dst)
    
    # Copy colmap_ba data
    colmap_ba_src = root_path / "post" / "sparse" / "offline" / "lightglue" / "final"
    colmap_ba_dst = colmap_dir / "lightglue" / "0"
    copy_directory(colmap_ba_src, colmap_ba_dst)

    # Copy colmap_ba data
    colmap_ba_src = root_path / "post" / "sparse" / "offline" / "loftr" / "final"
    colmap_ba_dst = colmap_dir / "loftr" / "0"
    copy_directory(colmap_ba_src, colmap_ba_dst)

    # Copy colmap_ba data
    colmap_ba_src = root_path / "post" / "sparse" / "offline" / "colmap" / "final"
    colmap_ba_dst = colmap_dir / "colmap" / "0"
    copy_directory(colmap_ba_src, colmap_ba_dst)
    
    # Copy glomap_ba data
    glomap_ba_src = root_path / "post" / "sparse" / "offline" / "glomap" / "final" / "0"
    glomap_ba_dst = colmap_dir / "glomap" / "0"
    copy_directory(glomap_ba_src, glomap_ba_dst)
    
    # Copy images
    images_src = root_path / "post" / "images"
    images_dst = output_root / "images"
    copy_directory(images_src, images_dst)

    print(f"Conversion completed successfully. Output directory: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARKit 3DGS output for nerfstudio training.")
    parser.add_argument("--input_path", help="Path to the root directory of run_arkit_3dgs.sh output")
    args = parser.parse_args()
    
    main(args)