import collections
import datetime as dt
import json
import math
import sqlite3

import cv2 as cv
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import datetime
import time
import math
# from hungarian_algorithm import algorithm
from munkres import Munkres, print_matrix

import sys


from GSMC.gsmc import project_point_cloud
import fine_grained_segmentation.datasets.dataset_configs as data_configs
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_next_bytes, qvec2rotmat, read_images_binary, read_cameras_binary
from GSMC.gsmc_utils import get_point_cloud_info
from pose_estimation.pipeline import get_ground_truth_poses, get_descriptors_image, get_camera_parameters, get_reference_images_info_binary
from visualization.visualization_tools import show_kp


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def project_points(Rmat, tvec, camera_internals, point_cloud_info, project_point_ids=None, height=768, width=1024, correct_distortion=False, distortion_coefs=None ):
    if project_point_ids is None:
        project_point_ids = [n for n in point_cloud_info.keys()]

    colors = []
    p3D = np.array([point_cloud_info[pid].xyz for pid in project_point_ids])
    if correct_distortion:
        correct_distortion = [0,0,0,0]
    p2D = cv.projectPoints(p3D, Rmat, tvec, camera_internals, np.array(distortion_coefs))[0].reshape(
        (len(project_point_ids), 2))

    for pid in project_point_ids:
        colors.append(point_cloud_info[pid].rgb)

    #
    # X_hcoords = np.ones((4, len(project_point_ids)))
    # for c, pid in enumerate(project_point_ids):
    #     X_hcoords[:, c] = np.array(list(point_cloud_info[pid].xyz) + [1])
    #     colors.append(point_cloud_info[pid].rgb)
    #
    # x_hcoords = np.matmul(P, X_hcoords)
    # x_hcoords = x_hcoords / x_hcoords[2, :]
    # if correct_distortion:
    #     x = x_hcoords[0, :]
    #     y = x_hcoords[1, :]
    #     r2 = x ** 2 + y ** 2
    #     x_d = x * (1 + distortion_coefs[0] * r2 + distortion_coefs[1] * r2 ** 2) + 2 * distortion_coefs[2] * x * y + \
    #           distortion_coefs[3] * (r2 + 2 * x ** 2)
    #     y_d = y * (1 + distortion_coefs[0] * r2 + distortion_coefs[1] * r2 ** 2) + 2 * distortion_coefs[3] * x * y + \
    #           distortion_coefs[2] * (r2 + 2 * y ** 2)
    #     x_hcoords[0, :] = x_d
    #     x_hcoords[1, :] = y_d
    # x_hcoords = np.matmul(camera_internals, x_hcoords)
    x_coords_effective = []
    colors_effective = []
    pids_effective = []

    for c, pid in enumerate(project_point_ids):
        if 0 < round(p2D[c,0]) < width and 0 < round(p2D[c,1]) < height:
            x_coords_effective.append(p2D[c, :])
            colors_effective.append(colors[c])
            pids_effective.append(pid)

    return np.array(x_coords_effective), colors_effective, pids_effective

def get_correspondence(corr_name):
    corr_set_config = data_configs.CmuConfig()

    # open correspondence file
    mat_content = {}
    ff = h5py.File(os.path.join(
        corr_set_config.correspondence_path, corr_name), 'r')
    for k, v in ff.items():
        mat_content[k] = np.array(v)

    return mat_content


def check_corresponding_image_counts(slicepath):
    corr_set_config = data_configs.CmuConfig()
    slice = '22'

    ref_names = dict()
    q_names = dict()
    mat = []
    for root, subdir, file in os.walk(corr_set_config.correspondence_path):
        for f in file:
            if 'slice' + slice in f and f.endswith('.mat'):
                mat_content = get_correspondence(f)

                # get image names
                ref_name = ''.join(chr(a) for a in mat_content['im_i_path']).split('/')[2]
                query_name = ''.join(chr(a) for a in mat_content['im_j_path']).split('/')[2]

                ref_names[ref_name] = ref_names.get(ref_name, 0) + 1
                if ref_names[ref_name] == 1:
                    q_names[ref_name] = [(query_name,f)]
                else:
                    q_names[ref_name] += [(query_name,f)]
                # for plotting correspondences among 2 imgs only
                if ref_name == 'img_07652_c1_1285950332288146us.jpg' and query_name == 'img_07899_c1_1292959625065025us.jpg':
                    mat = mat_content
                # if ref_name == 'img_07652_c1_1285950332288146us.jpg':
                #     mat.append(mat_content)

    for k, v in ref_names.items():
        if v==2:
            print(k)
            print(q_names[k])

    return mat


# def exploratory_analysis():
#     slice_path = '/home/valentinas98/repos/Thesis_repo/Data/Extended-CMU-Seasons/slice22'
#     slice = '22'
#
#     corr_mat = check_corresponding_image_counts(slice_path)
#
#     # choose some image names to perform check. We choose the ones with most corresponding images? Or that's a bias?
#     # query_names = ['img_07721_c0_1285950338287044us.jpg', 'img_07613_c0_1285950328888719us.jpg']
#     # for now only one
#     # query_names = ['img_07595_c1_1284563914987092us.jpg'] #['img_07253_c1_1303399116249565us.jpg'] # 'img_07613_c0_1285950328888719us.jpg' # no correspondences here... but I used it for some analysis (projections)
#     query_names = ['img_08364_c1_1288793059850196us.jpg']  # ['img_07899_c1_1292959625065025us.jpg']
#     ref_name = 'img_07652_c1_1285950332288146us.jpg'
#
#     # get point cloud info
#     full_points3D = get_point_cloud_info(slice_path)
#
#     # get further point cloud information for additional point filtering
#     database_path = '/home/valentinas98/repos/Thesis_repo/Data/Extended-CMU-Seasons/slice' + str(
#         slice) + '/database' + str(slice) + '.db'
#     connection = sqlite3.connect(database_path)
#     cursor = connection.cursor()
#
#     imagesbin_path = '/home/valentinas98/repos/Thesis_repo/Data/Extended-CMU-Seasons/slice' + str(
#         slice) + '/sparse/images.bin'
#     db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names, cursor = get_reference_images_info_binary(
#         imagesbin_path, cursor)
#
#     original_des_size = len(db_p3D_ids)
#
#     db_image_ids = [i for c, i in enumerate(db_image_ids) if db_p3D_ids[c] != -1]
#     db_image_names = [i for c, i in enumerate(db_image_names) if db_p3D_ids[c] != -1]
#     db_kp_coords = [(db_kp_coords_x[i], db_kp_coords_y[i]) for i in range(len(db_kp_coords_x)) if
#                     db_p3D_ids[i] != -1]  # hay problema
#     db_descriptors = db_descriptors[[c for c, i in enumerate(db_p3D_ids) if i != -1], :]
#     db_p3D_ids = [i for c, i in enumerate(db_p3D_ids) if db_p3D_ids[c] != -1]
#     db_camera_id = ['c0' if 'c0' in name else 'c1' for name in db_image_names]
#
#     # get all queries GT
#     poses, query_names = get_ground_truth_poses(query_names, slice_path)
#     # region Database queries
#     # with open(os.path.join(slice_path, 'ground-truth-database-images-slice22.txt'), 'r') as f:
#     #     lines = f.readlines()
#     #     lines = [l.split(' ') for l in lines]
#     #     poses = [[float(t) for t in p[1:8]] for p in lines if p[0] in query_names]
#     #     query_names = [p[0] for p in lines if p[0] in query_names]
#     # endregion
#
#     # region To use when want to check database images and not queries
#     poses = []
#     image_ids = []
#     kp_coords_x = []
#     kp_coords_y = []
#     point3Ds = []
#     image_names = []
#     with open(imagesbin_path, "rb") as fid:
#         num_reg_images = read_next_bytes(fid, 8, "Q")[0]
#         for _ in range(num_reg_images):
#             binary_image_properties = read_next_bytes(
#                 fid, num_bytes=64, format_char_sequence="idddddddi")
#             image_id = binary_image_properties[0]
#             poses.append(binary_image_properties[1:8])
#             current_char = read_next_bytes(fid, 1, "c")[0]
#             image_name = ""
#             while current_char != b"\x00":  # look for the ASCII 0 entry
#                 image_name += current_char.decode("utf-8")
#                 current_char = read_next_bytes(fid, 1, "c")[0]
#
#             num_points2D = read_next_bytes(fid, num_bytes=8,
#                                            format_char_sequence="Q")[0]
#             x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
#                                        format_char_sequence="ddq" * num_points2D)
#             xs = list(map(float, x_y_id_s[0::3]))
#             ys = list(map(float, x_y_id_s[1::3]))
#             point3D_ids = list(map(int, x_y_id_s[2::3]))
#             image_ids = image_ids + [image_id for i in range(num_points2D)]
#             image_names = image_names + [image_name for i in range(num_points2D)]
#             kp_coords_x = kp_coords_x + xs
#             kp_coords_y = kp_coords_y + ys
#             point3Ds = point3Ds + point3D_ids
#     # endregion
#
#     for index in range(len(query_names)):
#         # get image keypoints,
#         qkp, q_descriptors = get_descriptors_image(query_names[index], slice_path, 'query', slice=slice)
#         qkp = [[kp[0], kp[1]] for kp in qkp]
#
#         rkp, r_descriptors = get_descriptors_image(ref_name, slice_path, 'query', slice=slice)
#         rkp = [[kp[0], kp[1]] for kp in rkp]
#         # region Database stuff
#         # qkp = [ db_kp_coords[c] for c, name in enumerate(db_image_names) if db_p3D_ids[c]!=-1 and name==query_names[index]]
#         # endregion
#
#         # region for database images: verify that the point cloud is keeping right points
#         # point3D_path = slice_path + '/sparse/points3D.bin'
#         # imgs = read_images_binary(imagesbin_path)
#         # pts3D = read_points3D_binary(point3D_path)
#         #
#         # image_data = []
#         # for img in imgs.values():
#         #     if img.name == query_names[index]:
#         #         image_data = img
#         #         break
#         #
#         # seen_points_ids = image_data.point3D_ids
#         # seen_points = [list(pts3D[idx].xyz) for idx in seen_points_ids if idx != -1]
#         # endregion
#
#         # get intrinsics and distortion info
#         camera_id = 'c0' if 'c0' in query_names[index] else 'c1'
#         intrinsics, dist_coefs = get_camera_parameters(camera_id)
#
#         # get image ground truth pose,
#         r_gt = poses[index][:4]
#         R_gt = qvec2rotmat(r_gt)
#         c_gt = np.array(poses[index][4:])
#
#         # and ground truth P (without K)
#         t = - R_gt.dot(c_gt)
#         # c_gt = - R_gt.T.dot(t)
#         P = np.zeros((3, 4))
#         P[:, :-1] = R_gt
#         P[:, 3] = t
#
#         # project points from right camera only
#         # proj_ids = set()
#         # for c, db_p3D_id in enumerate(db_p3D_ids):
#         #     if db_camera_id[c] == camera_id:
#         #         proj_ids.update([db_p3D_id])
#         # proj_ids = [k for k in full_points3D.keys() if k in proj_ids]
#         # proj_qkp, proj_colors, proj_pids = project_points(P, intrinsics, full_points3D, project_point_ids=proj_ids, correct_distortion=True, distortion_coefs=dist_coefs)
#
#         # region Database plot
#         # fig = plt.figure()
#         # ax = plt.axes(projection='3d')
#         # ax.scatter3D([pt[0] for pt in seen_points], [pt[1] for pt in seen_points], [pt[2] for pt in seen_points], c=np.array(['fuchsia' for idx in seen_points_ids if idx != -1]), s=1)
#         # wannabe_pts = [list(pts3D[idx].xyz) for idx in proj_pids]
#         # ax.scatter3D([pt[0] for pt in wannabe_pts], [pt[1] for pt in wannabe_pts], [pt[2] for pt in wannabe_pts], c='cyan', s=1)
#         # plt.show()
#         # endregion
#
#         # there's a problem of visibility. I think we should try 2 ways:
#         # 1. use the point visibility information to restrict the group consciously ( and check how many remain compared to the gt )
#         # 2. use the viewed points of the closest image in the database (we know it since we have the gt centers)
#         # which one? 2. is probably faster but approximated and biased towards what's seen in the reference img.
#         # It's also true that probably 1. is also biased and anyway the point cloud was reconstructed from there.
#         visibility_info_pids = []
#         # for pid in proj_pids:
#         #     pt = full_points3D[pid]
#         #     dif = (c_gt - pt.xyz)
#         #     ssd = np.sum(dif ** 2)
#         #     cos_angle = dif.dot(pt.v) / math.sqrt(ssd)
#         #     if pt.dlow ** 2 -1 < ssd < pt.dup ** 2+1 and np.arccos(cos_angle)<pt.theta/2*1.05:
#         #         visibility_info_pids.append(pid)
#
#         # region Database plot
#         # fig = plt.figure()
#         # ax = plt.axes(projection='3d')
#         # ax.scatter3D([pt[0] for pt in seen_points], [pt[1] for pt in seen_points], [pt[2] for pt in seen_points],
#         #              c=np.array(['fuchsia' for idx in seen_points_ids if idx != -1]), s=1)
#         # wannabe_pts = [list(pts3D[idx].xyz) for idx in visibility_info_pids]
#         # ax.scatter3D([pt[0] for pt in wannabe_pts], [pt[1] for pt in wannabe_pts], [pt[2] for pt in wannabe_pts],
#         #              c='cyan', s=1)
#         # plt.show()
#         # endregion
#
#         # Plot projected points against keypoints first.
#         # show_kp(os.path.join(slice_path, 'query/'+query_names[index]), qkp + list(proj_qkp), col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(proj_qkp))], alpha=0.8)
#         # show_kp(os.path.join(slice_path, 'query/'+query_names[index]), list(proj_qkp), col=np.array([rgb_to_hex(proj_color) for proj_color in proj_colors]))
#         # va bene dai la compro - confronta con img img_07288_c1_1303399120716257us.jpg
#
#         # region Database trials
#         # show_kp(os.path.join(slice_path, 'database/' + query_names[index]), qkp + list(proj_qkp),
#         #         col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(proj_qkp))], alpha=0.8)
#         # endregion
#
#         # region Visualization 2 - altra visualizzazione: i punti che sono in corrispondenza (rispetto al totale dei keypoints e rispetto ai punti proiettati
#         # Mi aspetto che siano più vicini ai punti proiettati
#         # show_kp(os.path.join(slice_path, 'query/'+query_names[index]), qkp + corr_mat['pt_j'].tolist(), col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(corr_mat['pt_j']))])
#         show_kp(os.path.join(slice_path, 'query/' + ref_name), rkp + corr_mat['pt_i'].tolist(),
#                 col=['fuchsia' for _ in range(len(rkp))] + ['cyan' for _ in range(len(corr_mat['pt_i']))])
#
#         # endregion
#
#     # region Visualization 3 - altra ancora: reference su cui plotti tutti i punti delle corrispondenze
#
#     # plot_kps = []
#     # colors = []
#     # col_list = ['orangered', 'yellow', 'cyan', 'navy', 'darkviolet', 'crimson', 'lawngreen', 'hotpink', 'white']
#     # for c, d in enumerate(corr_mat):
#     #     plot_kps += d['pt_i'].tolist()
#     #     colors += [col_list[c] for _ in range(len(d['pt_i']))]
#     #
#     # show_kp(os.path.join(slice_path, 'query/'+ref_name),plot_kps, col=colors, alpha=0.8 )
#
#     # endregion
#
#     #######################
#
#     # 1. find spatially admissible neighbors
#     # round of assignments: search NNs of projected 3D pts within a circle of certain ray.
#     # Add fictitious point that
#     # represents no assignment -> all must be assigned, all circles have at least 2 options or if only one the point
#     # in the PC has no correspondent
#
#     # temporary assignments: every PC point assigned to its nearest neighbor in the image if any. If any two PC points are NN of the same point the match is established with the closest
#
#     # Get corresponding image keypoints. Repeat projection there
#
#     # Capire in che formato sono le corrispondenze e i keypoint (pixel interi o float). Capire se aggiungere corrispondenze può aiutare o confonde


def get_gt_matches(query_name, slicepath, slice, qkp=None, full_points3D=None, visibility_check=True, return_prjpts=False, proj_pts_dict=None):

    # region (1) Load data:
    if full_points3D is None:
        full_points3D = get_point_cloud_info(slicepath)

    pose_gt_query = get_ground_truth_poses([query_name], slicepath)[0][0]

    if qkp is None:
        qkp, q_descriptors = get_descriptors_image(query_name, slicepath, 'query', slice=slice)
        qkp = [[kp[0], kp[1]] for kp in qkp]

    r_gt_query = np.array(pose_gt_query[:4])
    R_gt_query = qvec2rotmat(r_gt_query)
    c_gt_query = np.array(pose_gt_query[4:])
    t_gt_query = - R_gt_query.dot(c_gt_query)

    images_path = slicepath + '/sparse/images.bin'
    img_data = read_images_binary(images_path)
    camera_path = slicepath + '/sparse/cameras.bin'
    # endregion

    # region (2) Select points in the point cloud to project:
    # visibility filter
    camera_id = 'c0' if 'c0' in query_name else 'c1'
    intrinsics, dist_coefs = get_camera_parameters(camera_id)

    camera_code = 1 if camera_id == 'c0' else 2

    if proj_pts_dict is None or not visibility_check:
        proj_ids = set()
        for im in img_data.values():
            if im.camera_id == camera_code:
                proj_ids.update(im.point3D_ids)
        proj_ids = [k for k in full_points3D.keys() if k in proj_ids]


    visibility_info_pids = []

    if visibility_check and proj_pts_dict is not None:
        dif = (c_gt_query - proj_pts_dict['coord'])
        ssd = np.sum((c_gt_query - proj_pts_dict['coord'])**2, axis=1)
        cos_angles = np.sum(proj_pts_dict['mean_v_dirs'] * dif,
                            axis=1) / np.sqrt(ssd)
        pid_mask = np.greater(cos_angles, np.cos(proj_pts_dict['thetas'] / 2))
        partial_mask = np.logical_and(np.greater(ssd, (0.9 * proj_pts_dict['dlow']) ** 2), np.greater((1.1 * proj_pts_dict['dup']) ** 2, ssd))
        pid_mask = np.logical_and(pid_mask, partial_mask)

        if np.sum(pid_mask)==0:
            if return_prjpts:
                return [], [], []
            return np.zeros(shape=(len(qkp), len(full_points3D.keys())), dtype=np.float64)
        p2D = cv.projectPoints(proj_pts_dict['coord'][pid_mask], cv.Rodrigues(R_gt_query)[0], t_gt_query, intrinsics, np.array(dist_coefs))[0].reshape((np.sum(pid_mask), 2))
        p2D = np.rint(p2D).astype(np.int16)
        within_range = np.logical_and(np.greater(p2D, np.zeros_like(p2D)), np.greater(
            np.append(np.ones((len(p2D), 1)) * 1024, np.ones((len(p2D), 1)) * 768, axis=1), p2D)).all(axis=1)

        proj_qkp = p2D[within_range]
        proj_qpids = proj_pts_dict['pts_ids'][pid_mask][within_range]


    elif visibility_check and proj_pts_dict is None:
        for pid in proj_ids:
            pt = full_points3D[pid]
            dif = (c_gt_query - pt.xyz)
            ssd = np.sum(dif ** 2)
            cos_angle = dif.dot(pt.v) / math.sqrt(ssd)
            if (pt.dlow*0.9) ** 2 <= ssd <= (pt.dup*1.1) ** 2 and np.arccos(cos_angle)<=pt.theta/2:
                visibility_info_pids.append(pid)

        if len(visibility_info_pids) == 0:
            if return_prjpts:
                return [], [], []
            return np.zeros(shape=(len(qkp), len(full_points3D.keys())), dtype=np.float64)
        proj_qkp, proj_qcolors, proj_qpids = project_points(R_gt_query, t_gt_query, intrinsics, full_points3D,
                                                            project_point_ids=visibility_info_pids,
                                                            correct_distortion=True, distortion_coefs=dist_coefs)


    else:

        # closest_db_img = ''
        # closest_dist = 100000
        img_centers = []
        img_names = []
        with open(os.path.join(slicepath, 'ground-truth-database-images-slice' + slice + '.txt'), 'r') as f:
            for l in f.readlines():
                if camera_id in l.split()[0]:
                    img_centers.append([float(x) for x in l.split()[5:8]])
                    img_names.append(l.split()[0])
                # if np.linalg.norm(np.array([float(x) for x in l.split()[5:8]])-c_gt_query) < closest_dist and camera_id in l.split()[0]:
                #     closest_dist = np.linalg.norm(np.array([float(x) for x in l.split()[5:8]])-c_gt_query)
                #     closest_db_img = l.split()[0]
        img_names = np.array(img_names)
        img_centers = np.array(img_centers)

        all_dists = distance_matrix([c_gt_query], np.array(img_centers)).flatten()
        dists_sort_idx = np.argsort(all_dists)
        all_dists = all_dists[dists_sort_idx]
        closest_db_img = img_names[dists_sort_idx][0]

        add_more = True
        if add_more:
            close_imgs = img_names[dists_sort_idx][1:11]
            close_imgs = close_imgs[np.where(all_dists[1:11]<10)]

        visibility_info_pids = set()

        for img in img_data.values():
            if img.name == closest_db_img or (add_more and img.name in close_imgs):
                visibility_info_pids = visibility_info_pids.union([pt for pt in img.point3D_ids if pt!=-1])




        if len(visibility_info_pids)==0:
            if return_prjpts:
                return [], [], []
            return np.zeros(shape=(len(qkp), len(full_points3D.keys())), dtype=np.float64)
        proj_qkp, proj_qcolors, proj_qpids = project_points(R_gt_query, t_gt_query, intrinsics, full_points3D, project_point_ids=visibility_info_pids,
                                                        correct_distortion=True, distortion_coefs=dist_coefs)

    verified_proj_qkps = proj_qkp
    verified_projpt_ids = proj_qpids
    if len(verified_projpt_ids) == 0:
        if return_prjpts:
            return visibility_info_pids, [], []
        return np.zeros(shape=(len(qkp), len(full_points3D.keys())), dtype=np.float64)
    # endregion

    # region (3) Select keypoints in a neighborhood of each remaining point

    kp_dist_matrix = distance_matrix(qkp, verified_proj_qkps)
    kp_exp_matrix = np.exp(-kp_dist_matrix)

    if return_prjpts:
        return visibility_info_pids, verified_projpt_ids, kp_exp_matrix
    # endregion

    # region (4) Build matches binary matrix
    match_gt_matrix = np.zeros(shape=(len(qkp), len(full_points3D.keys())), dtype=np.float64)
    for c, pid in enumerate(full_points3D.keys()):
        if pid in verified_projpt_ids:
            idx = list(verified_projpt_ids).index(pid)
            match_gt_matrix[:, c] = kp_exp_matrix[:,idx]
            # endregion

    # region (6) (Optional) Visualize the original projected points and left points to make sure they are ok
    # show_kp(os.path.join(slice_path, 'query/'+query_name), qkp + list(proj_qkp), col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(proj_qkp))], alpha=0.8)
    # show_kp(os.path.join(slice_path, 'query/'+query_name), qkp + list(verified_proj_qkps), col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(verified_proj_qkps))], alpha=0.9 )
    # print(len(verified_proj_qkps))
    # show_kp(os.path.join(slice_path, 'query/'+query_name), qkp + list(q_corr), col=['fuchsia' for _ in range(len(qkp))] + ['cyan' for _ in range(len(q_corr))], alpha=0.8)
    # # endregion
    return match_gt_matrix


def esperimenti_vari_di_img_con_filtro_corrispondenze(slicepath, slice):
    # Prima immagine che ho studiato. Le decisioni potrebbero avere bias qui. Scelta per avere tante img corrispondenti, per il resto a caso
    query_name = 'img_07899_c1_1292959625065025us.jpg'
    correspondence_name = 'correspondence_slice22_run13_run29_c1362_c2346.mat'
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)

    # Voglio provare questa img perchè ricordo che in questa view c'erano nuvole ricostruite in 3D che si proiettavano. Si spera che le corrispondenze le tolgano
    # non si vede molto quello che cercavo
    query_name = 'img_07405_c0_1290445322611210us.jpg'
    correspondence_name = 'correspondence_slice22_run13_run28_c1194_c2192.mat'
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)

    # riproviamo
    query_name = 'img_07858_c0_1292959621598342us.jpg'
    correspondence_name = 'correspondence_slice22_run13_run29_c1275_c2263.mat'
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)

    # check di una img dove ci sono solo 2 corrispondenze
    # query_name/ = 'img_07572_c1_1288106907370711us.jpg'
    # correspondence_name = 'correspondence_slice22_run13_run25_c1328_c2360.mat'
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)

    # check di un caso in cui l'immagine è tutta rovinata
    query_name = 'img_07834_c0_1289590322289997us.jpg'
    correspondence_name = ''
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)

    query_name = 'img_07595_c1_1284563914987092us.jpg'
    correspondence_name = ''
    # get_gt_matches(query_name, correspondence_name, slicepath, slice)


    query_name = 'img_07595_c1_1284563914987092us.jpg'
    correspondence_name = ''
    get_gt_matches(query_name, slicepath, slice)


if __name__=='__main__':
    slice_path = '/home/valentinas98/repos/Thesis_repo/Data/Extended-CMU-Seasons/slice22'
    slice = '22'

    # Option 0: choose correspondence
    # check_corresponding_image_counts(slice_path)

    # Option 1: explore the placement of keypoints, point cloud points and correspondences
    # exploratory_analysis()

    # Option 2: get the "ground truth" correspondence matrix
    correspondence_name = 'correspondence_slice22_run13_run29_c1362_c2346.mat'
    query_name = 'img_07899_c1_1292959625065025us.jpg'

    # mat = get_gt_matches(query_name, correspondence_name, slice_path, slice)
    esperimenti_vari_di_img_con_filtro_corrispondenze(slice_path, slice)