import collections
import datetime as dt
import json
import math
import sqlite3
import time

import cv2 as cv
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1407)
import os
import pandas as pd
import torch
from scipy.spatial import distance_matrix
import datetime

from fine_grained_segmentation.utils.file_parsing.read_write_model import read_next_bytes


def get_descriptors_image(image_name, slice_path, type, slice):
    """Returns keypoint and descriptor lists for the specified image, querying the database"""

    database_path = os.path.join(slice_path,  type + str(slice) + '.db')
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT image_id FROM images WHERE name=?;",
                   (image_name,))

    for row in cursor:
        image_id = row[0]
        cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                       (image_id,))

        row = next(cursor)
        if row[0] is None:
            keypoints = np.zeros((0, 4), dtype=np.float32)
            descriptors = np.zeros((0, 128), dtype=np.uint8)
        else:
            keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)
            cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
                           (image_id,))
            row = next(cursor)
            descriptors = np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)

        # keypoints: coord 0 e 1 are respectively x (------>) and y (|
        #                                                            |
        #                                                            V)
        # starting from top-left (0,0) and with every pixel increasing by 1
        # additional 4 numbers represent with an affinity the direction and orientation of every feature

        # keypoints and descriptors share the same order
        return keypoints, descriptors
    return [], []


def get_camera_parameters(camera_id):
    '''Utility to get camera parameters, as from CMU-Extended-Seasons specifications'''
    intrinsics = np.zeros((3,3))

    if camera_id == 'c0':
        intrinsics[0][0] = 868.993378
        intrinsics[1][1] = 866.063001
        intrinsics[0][2] = 525.942323
        intrinsics[1][2] = 420.042529
        intrinsics[2][2] = 1

        k1 = -0.399431
        k2 = 0.188924

        p1 = 0.000153
        p2 = 0.000571
    else:
        intrinsics[0][0] = 873.382641
        intrinsics[1][1] = 876.489513
        intrinsics[0][2] = 529.324138
        intrinsics[1][2] = 397.272397
        intrinsics[2][2] = 1

        k1 = -0.397066
        k2 = 0.181925

        p1 = 0.000176
        p2 = -0.000579

    distortion_coefs = [k1, k2, p1, p2]

    return intrinsics, distortion_coefs


def get_reference_images_info_binary(images_file, cur):
    '''Utility to read reference images data'''
    image_ids = []
    kp_coords_x = []
    kp_coords_y = []
    point3Ds = []
    descriptors = np.empty((0, 128), dtype=np.uint8)
    image_names = []

    with open(images_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xs = list(map(float, x_y_id_s[0::3]))
            ys = list(map(float, x_y_id_s[1::3]))
            point3D_ids = list(map(int, x_y_id_s[2::3]))
            image_ids = image_ids + [image_id for i in range(num_points2D)]
            image_names = image_names + [image_name for i in range(num_points2D)]
            kp_coords_x = kp_coords_x + xs
            kp_coords_y = kp_coords_y + ys
            point3Ds = point3Ds + point3D_ids
            cur.execute("SELECT data FROM descriptors WHERE image_id=?;",
                        (image_id,))
            row = next(cur)
            descriptors = np.concatenate((descriptors, np.fromstring(row[0], dtype=np.uint8).reshape(-1, 128)), axis=0)

    return image_ids, kp_coords_x, kp_coords_y, point3Ds, descriptors, image_names, cur


def get_ground_truth_poses(query_names, slicepath):
    '''Utility to get the ground truth poses for selected queries'''
    pose_dict = dict()
    for root, subdir, files in os.walk(os.path.join(slicepath, 'camera-poses')):
        for f in files:
            with open(os.path.join(root, f)) as file:
                lines = file.readlines()
                for l in lines:
                    l = l.split()
                    pose_dict[l[0]] = [float(i) for i in l[1:]]

    pose_list = []
    discarded_queries = []
    for c, q in enumerate(query_names):
        try:
            pose_list.append(pose_dict[q])
        except:
            print("Query %s was discarded" %(q))
            discarded_queries.append(c)
    print("%d images discarded"%(len(discarded_queries)))
    return pose_list, [q for c, q in enumerate(query_names) if c not in discarded_queries]


# def get_gt_centres_trajectory(slicepath, slice, camera_id):
#
#     with open(os.path.join(slicepath, 'ground-truth-database-images-slice'+ slice +'.txt'), 'r') as f:
#         lines = f.readlines()
#         lines = [l.split(' ') for l in lines]
#         centres = [[float(t) for t in p[5:8]] for p in lines]
#         ref_names = [p[0] for p in lines]
#
#     return [centres[i] for i in range(len(ref_names)) if camera_id in ref_names[i]], [r_name for r_name in ref_names if camera_id in ref_names]
