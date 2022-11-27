import collections
import datetime as dt
import json
import math
import sqlite3
import time

import cv2 as cv
import numpy as np

np.random.seed(2250)
import os
import pandas as pd
import datetime

import context

from pose_estimation.utils.data_loading import get_reference_images_info_binary, get_descriptors_image, \
    get_ground_truth_poses, get_camera_parameters
from GSMC.gsmc_utils import get_point_cloud_info, qvec2rotmat
from GSMC.gsmc import GSMC_score
from experiments.analysis_matching import load_data, k_ratio_test
from pose_estimation.utils.eval import compute_pose_errors
from pose_estimation.RANSAC_custom import p3p_biased_RANSAC, p3p_robust_biased_sampling_and_consensus
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary


def get_stats(query_names, k, slicepath, slice, savepath, ratio_threshold=0.2, height=768, width=1024, window=15, largest_score=False):
    """
    Comparison of variations of our strategy. We include
    - a version with k=2
    - a version with k=1
    - a version with small window (5x5)
    - a version with large window (15x15)
    - a version with largest score only
    - a version with all scores>threshold
    In another exp we will try the other score and see if it makes a difference
    """

    stats_df = pd.DataFrame([], columns=['img_name',  'experiment_type','k', 'matches_in', 'matches_in_effective', 'success', 'inliers', 'position_error', 'orientation_error'])

    # Load database information only once
    database_path = slicepath + '/database' + str(slice) + '.db'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    imagesbin_path = slicepath + '/sparse/images.bin'
    db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names, cursor = get_reference_images_info_binary(
        imagesbin_path, cursor)

    db_descriptors = db_descriptors[[c for c, i in enumerate(db_p3D_ids) if i != -1], :]
    db_p3D_ids = [i for c, i in enumerate(db_p3D_ids) if db_p3D_ids[c] != -1]

    full3Dpoints = get_point_cloud_info(slicepath)
    all_pids = np.array(list(full3Dpoints.keys()))
    all_p3D = np.array([x.xyz for x in full3Dpoints.values()])

    images_path = slicepath + '/sparse/images.bin'
    img_data = read_images_binary(images_path)

    for q_num, query_name in enumerate(query_names):
        print("Doing query %d/%d" %(q_num, len(query_names)))
        # region (1) Load data
        data_dict = load_data(query_name, slicepath, slice, load_database=False)
        camera_id = data_dict['camera_id']
        camera_matrix = data_dict['camera_matrix']
        dist_coefs = data_dict['dist_coefs']
        kp_priority = data_dict['kp_priority']
        qkp = data_dict['qkp']
        q_descriptors = data_dict['qdesc']
        query_mask = data_dict['query_mask']
        c_gt = data_dict['c_gt']
        R_gt = data_dict['R_gt']
        g_direction = data_dict['g_direction']
        # centres_traj = data_dict['centres_traj']
        z0 = c_gt[2]
        # db_kp_3Dcoords = [full3Dpoints[p_id].xyz for p_id in db_p3D_ids]
        # q_kp_2Dcoords = [kp[:2] for kp in qkp]
        # endregion

        # region (2) Set up matches
        flann_matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        kplus1_nearest_matches = flann_matcher.knnMatch(q_descriptors.astype(np.float32),
                                                        db_descriptors.astype(np.float32),
                                                        k=k + 1)

        gsmc_scores = 0
        matches_kNN_5x5_all = []
        matches_kNN_15x15_all = []
        matches_kNN_5x5_largest = []
        matches_kNN_15x15_largest = []
        matches_1NN_15x15 = []

        all_ratios_kNN_15x15_all = []
        all_ratios_kNN_15x15_largest = []
        ratios_1NN_15x15 = []
        all_ratios_kNN_5x5_all = []
        all_ratios_kNN_5x5_largest = []

        for qkp_idx in kp_priority:

            m = kplus1_nearest_matches[qkp_idx][:-1]
            ratios = []
            is5x5 = []

            for neighbor_idx, putative_match in enumerate(m):

                qkp_x = min(round(qkp[qkp_idx][0]), width-1)
                qkp_y = min(round(qkp[qkp_idx][1]), height-1)

                sem = full3Dpoints[db_p3D_ids[putative_match.trainIdx]].semantic_label

                if (window < qkp_x < width - window and window < qkp_y < height-window and \
                        sem in query_mask[qkp_y - window:qkp_y + window,qkp_x - window:qkp_x + window]) or \
                        not( window < qkp_x < width - window and window < qkp_y < height-window):

                    score, _, _, totPointCounts, _ = GSMC_score(qkp[qkp_idx], db_p3D_ids[putative_match.trainIdx],
                                                                full3Dpoints, g_direction, camera_matrix,
                                                                dist_coefs, query_mask, slicepath, z0=z0, all_pids=all_pids, all_p3D=all_p3D)#, use_covisibility=True, img_data=img_data)


                    gsmc_scores += 1
                    if gsmc_scores % 100 == 0:
                        print("Computed %d scores " % (gsmc_scores))

                    if totPointCounts > 0: # and score / totPointCounts > ratio_threshold:
                        ratios.append(score) #/ totPointCounts)
                    else:
                        ratios.append(0)

                    if (5 < qkp_x < width - 5 and 5 < qkp_y < height - 5 and \
                        sem in query_mask[qkp_y - 5:qkp_y + 5, qkp_x - 5:qkp_x + 5]) or \
                            not (5 < qkp_x < width - 5 and 5 < qkp_y < height - 5):
                        is5x5.append(1)
                    else:
                        is5x5.append(0)

                else:
                    ratios.append(0)
                    is5x5.append(0)

            ratios = np.array(ratios)
            is5x5 = np.array(is5x5)

            top_idx = np.argmax(ratios)
            all_ratios_kNN_15x15_largest.append(ratios[top_idx])
            matches_kNN_15x15_largest.append(m[top_idx])

            matches_1NN_15x15.append(m[0])
            ratios_1NN_15x15.append(ratios[0])

            all_ratios_kNN_15x15_all += [r for r in ratios if r!=0]
            matches_kNN_15x15_all += [match for c, match in enumerate(m) if ratios[c] != 0]

            ratios[1-is5x5] = 0
            top_idx = np.argmax(ratios)
            all_ratios_kNN_5x5_largest.append(ratios[top_idx])
            matches_kNN_5x5_largest.append(m[top_idx]),

            all_ratios_kNN_5x5_all += [r for r in ratios if r != 0]
            matches_kNN_5x5_all += [match for c, match in enumerate(m) if ratios[c] != 0]

        # endregion


        # region (3) Perform RANSAC in its various flavors

        # set parameters
        reprojection_error = 8 # see https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp, function solvePnPRansac

        # unweighted
        ransac_dict = dict()
        ransac_dict['camera_matrix'] = camera_matrix
        ransac_dict['dist_coeff'] = dist_coefs



        if len(matches_1NN_15x15)>=4:
            p2D = np.array([qkp[m.queryIdx][:2] for m in matches_1NN_15x15])
            p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_1NN_15x15])
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D
            success_1NN_15x15, rvec_1NN_15x15, tvec_1NN_15x15, inliers_1NN_15x15 = p3p_biased_RANSAC(ransac_dict,
                                                                                                     ratios_1NN_15x15,
                                                                                                     10000,
                                                                                                     reprojection_error)

        else:
            success_1NN_15x15 = False
            inliers_1NN_15x15 = []
            rvec_1NN_15x15 = []

        # weighted
        if len(matches_kNN_15x15_largest) >= 4:
            ransac_dict['pts_2d'] = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_15x15_largest])
            ransac_dict['pts_3d'] = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_15x15_largest])

            success_kNN_15x15_largest, rvec_kNN_15x15_largest, tvec_kNN_15x15_largest, inliers_kNN_15x15_largest = p3p_biased_RANSAC(ransac_dict,
                                                                                                                                     all_ratios_kNN_15x15_largest, 10000, reprojection_error)
        else:
            success_kNN_15x15_largest = False
            inliers_kNN_15x15_largest = [0]
            rvec_kNN_15x15_largest = []


        if len(matches_kNN_15x15_all) >= 4:
            ransac_dict['pts_2d'] = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_15x15_all])
            ransac_dict['pts_3d'] = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_15x15_all])


            success_kNN_15x15_all, rvec_kNN_15x15_all, tvec_kNN_15x15_all, inliers_kNN_15x15_all = p3p_robust_biased_sampling_and_consensus(ransac_dict,
                                                                                                                                            all_ratios_kNN_15x15_all, 10000, reprojection_error)
        else:
            success_kNN_15x15_all = False
            inliers_kNN_15x15_all = [0]
            rvec_kNN_15x15_all = []

        if len(matches_kNN_5x5_largest) >= 4:
            ransac_dict['pts_2d'] = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_5x5_largest])
            ransac_dict['pts_3d'] = np.array(
                [full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_5x5_largest])

            success_kNN_5x5_largest, rvec_kNN_5x5_largest, tvec_kNN_5x5_largest, inliers_kNN_5x5_largest = p3p_biased_RANSAC(
                ransac_dict,
                all_ratios_kNN_5x5_largest, 10000, reprojection_error)
        else:
            success_kNN_5x5_largest = False
            inliers_kNN_5x5_largest = [0]
            rvec_kNN_5x5_largest = []

        if len(matches_kNN_5x5_all) >= 4:
            ransac_dict['pts_2d'] = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_5x5_all])
            ransac_dict['pts_3d'] = np.array(
                [full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_5x5_all])

            success_kNN_5x5_all, rvec_kNN_5x5_all, tvec_kNN_5x5_all, inliers_kNN_5x5_all = p3p_biased_RANSAC(
                ransac_dict,
                all_ratios_kNN_5x5_all, 10000, reprojection_error)
        else:
            success_kNN_5x5_all = False
            inliers_kNN_5x5_all = [0]
            rvec_kNN_5x5_all = []

        # endregion

        # save success, position error, rotation error, number of inliers, number of inliers that are real inliers

        if success_1NN_15x15 or len(rvec_1NN_15x15)>0:
            R_pred_1NN_15x15 = cv.Rodrigues(rvec_1NN_15x15)[0]
            c_pred_1NN_15x15 = - np.matmul(R_pred_1NN_15x15.T, tvec_1NN_15x15).flatten()
            position_error_1NN_15x15, rotation_error_1NN_15x15 = compute_pose_errors(R_gt, R_pred_1NN_15x15, c_gt,
                                                                                     c_pred_1NN_15x15)
        else:
            position_error_1NN_15x15 = -1
            rotation_error_1NN_15x15 = -1
            inliers_1NN_15x15 = []

        if success_kNN_15x15_largest or len(rvec_kNN_15x15_largest)>0:
            R_pred_kNN_15x15_largest = cv.Rodrigues(rvec_kNN_15x15_largest)[0]
            c_pred_kNN_15x15_largest = - np.matmul(R_pred_kNN_15x15_largest.T, tvec_kNN_15x15_largest).flatten()
            position_error_kNN_15x15_largest, rotation_error_kNN_15x15_largest = compute_pose_errors(R_gt, R_pred_kNN_15x15_largest, c_gt,
                                                                                                     c_pred_kNN_15x15_largest)
        else:
            position_error_kNN_15x15_largest = -1
            rotation_error_kNN_15x15_largest = -1
            inliers_kNN_15x15_largest = []

        if success_kNN_15x15_all or len(rvec_kNN_15x15_all)>0:
            R_pred_kNN_15x15_all = cv.Rodrigues(rvec_kNN_15x15_all)[0]
            c_pred_kNN_15x15_all = - np.matmul(R_pred_kNN_15x15_all.T, tvec_kNN_15x15_all).flatten()
            position_error_kNN_15x15_all, rotation_error_kNN_15x15_all = compute_pose_errors(R_gt, R_pred_kNN_15x15_all, c_gt, c_pred_kNN_15x15_all)
        else:
            position_error_kNN_15x15_all = -1
            rotation_error_kNN_15x15_all = -1
            inliers_kNN_15x15_all = []

        if success_kNN_5x5_largest or len(rvec_kNN_5x5_largest) > 0:
            R_pred_kNN_5x5_largest = cv.Rodrigues(rvec_kNN_5x5_largest)[0]
            c_pred_kNN_5x5_largest = - np.matmul(R_pred_kNN_5x5_largest.T, tvec_kNN_5x5_largest).flatten()
            position_error_kNN_5x5_largest, rotation_error_kNN_5x5_largest = compute_pose_errors(R_gt,
                                                                                                     R_pred_kNN_5x5_largest,
                                                                                                     c_gt,
                                                                                                     c_pred_kNN_5x5_largest)
        else:
            position_error_kNN_5x5_largest = -1
            rotation_error_kNN_5x5_largest = -1
            inliers_kNN_5x5_largest = []

        if success_kNN_5x5_all or len(rvec_kNN_5x5_all) > 0:
            R_pred_kNN_5x5_all = cv.Rodrigues(rvec_kNN_5x5_all)[0]
            c_pred_kNN_5x5_all = - np.matmul(R_pred_kNN_5x5_all.T, tvec_kNN_5x5_all).flatten()
            position_error_kNN_5x5_all, rotation_error_kNN_5x5_all = compute_pose_errors(R_gt, R_pred_kNN_5x5_all,
                                                                                             c_gt, c_pred_kNN_5x5_all)
        else:
            position_error_kNN_5x5_all = -1
            rotation_error_kNN_5x5_all = -1
            inliers_kNN_5x5_all = []


        query_stats = pd.DataFrame()
        query_stats['img_name'] = [query_name for i in range(5)]
        query_stats['experiment_type'] = ['1NN_15x15', 'kNN_15x15_largest', 'kNN_15x15_all',  'kNN_5x5_largest', 'kNN_5x5_all' ]
        query_stats['k'] = [1, k, k, k, k]
        query_stats['matches_in'] = [len(matches_1NN_15x15), len(matches_kNN_15x15_largest), len(matches_kNN_15x15_all), len(matches_kNN_5x5_largest), len(matches_kNN_5x5_all)]
        query_stats['matches_in_effective'] = [len([s for s in ratios_1NN_15x15 if s > ratio_threshold]), len([s for s in all_ratios_kNN_15x15_largest if s > ratio_threshold]),
                                               len([m for m in all_ratios_kNN_15x15_all if m > ratio_threshold]),
                                               len([m for m in all_ratios_kNN_5x5_largest if m > ratio_threshold]),
                                               len([m for m in all_ratios_kNN_5x5_all if m > ratio_threshold])]
        query_stats['success'] = [success_1NN_15x15, success_kNN_15x15_largest, success_kNN_15x15_all, success_kNN_5x5_largest, success_kNN_5x5_all]
        query_stats['inliers'] = [sum(inliers_1NN_15x15), sum(inliers_kNN_15x15_largest), sum(inliers_kNN_15x15_all), sum(inliers_kNN_5x5_largest), sum(inliers_kNN_5x5_all)]
        query_stats['position_error'] = [position_error_1NN_15x15, position_error_kNN_15x15_largest, position_error_kNN_15x15_all, position_error_kNN_5x5_largest, position_error_kNN_5x5_all]
        query_stats['orientation_error'] = [rotation_error_1NN_15x15, rotation_error_kNN_15x15_largest, rotation_error_kNN_15x15_all, rotation_error_kNN_5x5_largest, rotation_error_kNN_5x5_all]

        stats_df = pd.concat([stats_df, query_stats], axis=0 )


    stats_df.to_csv(os.path.join(savepath, 'pose_est_stats.csv'))

    return stats_df


if __name__ == '__main__':
    # Example: exp on slice 22

    # Add your repo path
    my_repo_path = 'ADD YOUR REPO PATH'
    slicepath = my_repo_path + '/data/Extended-CMU-Seasons/slice22'
    slice = '22'

    k = 2

    with open(os.path.join(my_repo_path, 'pose_estimation/experiments/repeated_structures_queries_2022_08_22_18_23_57_c0_slice22_70imgs.txt'), 'r') as f:
        query_names = [n[:-1] for n in f.readlines()]

    now = str(datetime.datetime.now()).replace('-', '_').replace(':', '_').replace(' ', '_')[:-7]
    stats_dirname = f'exp_k{k}_n_queries{len(query_names)}_date{now}/'
    stats_dirname = os.path.join(my_repo_path, 'pose_estimation/results/semantic_matching_pose_comparisons',
                                 stats_dirname)


    if not os.path.exists(stats_dirname):
        os.mkdir(stats_dirname)

    get_stats(query_names, k, slicepath, slice, stats_dirname, largest_score=True)