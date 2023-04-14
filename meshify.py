#!/usr/bin/env/python3

"""
DESCRIPTION:
Make mesh object of CW labels and saving it to stl file.

"""

import numpy as np
import nibabel as nib
import time
import pandas as pd
from skimage import measure
from stl import mesh
from os.path import join
import sys

def parse_arguments():
    """
    Simple CommandLine argument parsing function making use of the argparse module

    :return: parsed arguments object args
    """
    parser = argparse.ArgumentParser(
        description=" Meshify object and save as stl file"
    )

    parser.add_argument(
        "-a",
        "--artery",
        help="Enter label number for target artery.",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-od",
        "--output_directory",
        help="Absolute path to output directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-cw",

        help="Labels of CW [ .nii / .nii.gz ]",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    return args


def make_mesh(image, spacing, threshold=-300, step_size=1):
    """
    Function that use the Marching Algorithm to create a 2D surface mesh from a 3D volume
    Input:
    image: 3D patient matrix
    spacing: Voxel sizes
    threshold: Threshold used by the Marching Cube to determine if a pixel is in the surface or not.
    step_size: Step size in voxels. Default 1. Larger steps yield faster but coarser results. The result will
               always be topologically correct though.
    Output:
    verts: Spatial coordinates for V unique mesh vertices. Coordinate order matches input volume (X, Y, Z).
    faces: Define triangular faces via referencing vertex indices from verts. This algorithm specifically outputs
               triangles, so each face has exactly three indices.
    """

    verts, faces, norm, val = measure.marching_cubes_lewiner(image, threshold, gradient_direction='descent',
                                                             spacing=spacing, step_size=step_size,
                                                             allow_degenerate=True)

    return verts, faces


def make_stl(verts, faces, stl_file_name='skull.stl', **kwargs):
    """ Function that make a STL file from vertices and triangle faces
    Input:
    verts: Spatial coordinates for V unique mesh vertices. Coordinate order matches input volume (X, Y, Z).
    faces: Triangular faces via referencing vertex indices from verts. Triangles, so each face has exactly three
           indices.
    stl_file_name: name of the STL file
    kwarg (optional): keyworded variable
    1) stl_path_name: path of the STL file
    """
    print('Making STL')
    skull = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            skull.vectors[i][j] = verts[f[j], :]

    if kwargs:
        stl_path_name = kwargs['stl_path_name']
        skull.save(join(stl_path_name, stl_file_name))


def main():
    # Argument Parsing
    args = parse_arguments()
    output_directory = args.output_directory
    artery = args.artery
    cw_path = args.cw

    # Load Images
    cw_img = nib.load(cw_path)

    # Extract data to numpy arrays
    cw = np.asanyarray(cw_img.dataobj).astype(float)

    # Filter artery
    if artery is not None:
        cw[cw != artery] = 0

    # Make Stl
    header = cw.header
    verts, faces = make_mesh(cw, header.get_zooms(), threshold=0.5, step_size=1)
    file = os.path.basename(file_path)
    file_name = os.path.splitext(file)
    output_file = file_name[0] + '.stl'
    make_stl(verts, faces, stl_file_name=output_file, stl_path_name=output_directory)


if __name__ == "__main__":
    main()
