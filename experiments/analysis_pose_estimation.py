# Crono versions:
# 16/08 before 21:50: version with all pose experiments with both our version of RANSAC and Opencv.
# 16/08 after 21:50: cleaning up to have only our RANSAC and adding SSMC. Ratio threshold from 0.2 to 0.001, semantic window 15 pixel
#                    seed 2250,
# 20/08 mezzogiorno: aggiunta la possibilità di usare solo i match col miglior ratio e valutazione dei modelli su tutti i match (compresi quelli con score zero)


import json
import sqlite3

import cv2 as cv
import numpy as np

np.random.seed(2250)
import os
import pandas as pd
import datetime

import sys

import context

from pose_estimation.utils.matching import ratio_test
from pose_estimation.utils.data_loading import get_reference_images_info_binary
from GSMC.gsmc_utils import get_point_cloud_info
from GSMC.gsmc import GSMC_score
from experiments.analysis_matching import load_data, k_ratio_test
from pose_estimation.utils.eval import compute_pose_errors
from pose_estimation.RANSAC_custom import p3p_biased_RANSAC, p3p_robust_biased_sampling_and_consensus
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary


def get_stats(query_names, k, slicepath, slice, savepath, ratio_threshold=0.2, height=768, width=1024, window=15, largest_score=False):
    stats_df = pd.DataFrame([], columns=['img_name',  'experiment_type','k', 'ratio_test', 'matches_in', 'matches_in_effective','success', 'inliers', 'position_error', 'orientation_error'])

    # Load database information only once
    database_path = slicepath + '/database' + str(slice) + '.db'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    imagesbin_path = slicepath + '/sparse/images.bin'
    db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names, cursor = get_reference_images_info_binary(
        imagesbin_path, cursor)

    images_path = slicepath + '/sparse/images.bin'
    img_data = read_images_binary(images_path)

    db_descriptors = db_descriptors[[c for c, i in enumerate(db_p3D_ids) if i != -1], :]
    db_p3D_ids = [i for c, i in enumerate(db_p3D_ids) if db_p3D_ids[c] != -1]

    full3Dpoints = get_point_cloud_info(slicepath)
    all_pids = np.array(list(full3Dpoints.keys()))
    all_p3D = np.array([x.xyz for x in full3Dpoints.values()])

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
        kplus1_nearest_matches = flann_matcher.knnMatch(q_descriptors.astype(np.float32), db_descriptors.astype(np.float32),
                                                        k=k + 1)

        matches_1NN_ratio09 = ratio_test([(m[0], m[1]) for m in kplus1_nearest_matches], 0.9)

        matches_kNN_ratio09 = k_ratio_test([m[:k + 1] for m in kplus1_nearest_matches], 0.9)
        matches_kNN_ratio09 = [m for mm in matches_kNN_ratio09 for m in mm]


        gsmc_scores = 0
        matches_kNN_semanticratio = []
        all_ratios = []

        matches_GSMC = []
        filtered_scores = []

        matches_SSMC = []

        for qkp_idx in kp_priority:

            m = kplus1_nearest_matches[qkp_idx][:-1]
            ratios = []


            for neighbor_idx, putative_match in enumerate(m):

                qkp_x = min(round(qkp[qkp_idx][0]), width-1)
                qkp_y = min(round(qkp[qkp_idx][1]), height-1)

                sem = full3Dpoints[db_p3D_ids[putative_match.trainIdx]].semantic_label

                if neighbor_idx == 0 and query_mask[qkp_y, qkp_x] == sem:
                    matches_SSMC.append(putative_match)

                if (window < qkp_x < width - window and window < qkp_y < height-window and \
                        sem in query_mask[qkp_y - window:qkp_y + window,qkp_x - window:qkp_x + window]) or \
                        not( window < qkp_x < width - window and window < qkp_y < height-window):

                    score, _, _, totPointCounts, _ = GSMC_score(qkp[qkp_idx], db_p3D_ids[putative_match.trainIdx],
                                                                full3Dpoints, g_direction, camera_matrix,
                                                                dist_coefs, query_mask, slicepath, z0=z0, all_pids=all_pids, all_p3D=all_p3D)

                    gsmc_scores += 1
                    if gsmc_scores % 100 == 0:
                        print("Computed %d scores " % (gsmc_scores))

                    if totPointCounts > 0 and score / totPointCounts > ratio_threshold:
                        ratios.append(score / totPointCounts)
                    else:
                        ratios.append(0)

                    if neighbor_idx == 0 and putative_match.distance < 0.9 * m[1].distance:
                            matches_GSMC.append(putative_match)
                            filtered_scores.append(score)


                else:
                    ratios.append(0)

                    if neighbor_idx == 0 and putative_match.distance < 0.9 * m[1].distance:
                        score, _, _, totPointCounts, _ = GSMC_score(qkp[qkp_idx],
                                                                    db_p3D_ids[putative_match.trainIdx],
                                                                    full3Dpoints, g_direction, camera_matrix,
                                                                    dist_coefs, query_mask, slicepath, z0=z0,
                                                                    all_pids=all_pids, all_p3D=all_p3D)#, use_covisibility=True,
                                                                    # img_data=img_data)
                        gsmc_scores += 1
                        if gsmc_scores % 100 == 0:
                            print("Computed %d scores " % (gsmc_scores))

                        matches_GSMC.append(putative_match)
                        filtered_scores.append(score)



            if largest_score:
                top_idx = np.argmax(ratios)
                all_ratios.append(ratios[top_idx])
                matches_kNN_semanticratio.append(m[top_idx])

            else:
                all_ratios += [r for r in ratios if r!=0]
                matches_kNN_semanticratio += [match for c, match in enumerate(m) if ratios[c]!=0]



        # endregion


        # region (3) Perform RANSAC in its various flavors

        # set parameters
        reprojection_error = 8 # see https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp, function solvePnPRansac

        # unweighted
        ransac_dict = dict()
        ransac_dict['camera_matrix'] = camera_matrix
        ransac_dict['dist_coeff'] = dist_coefs

        if len(matches_1NN_ratio09)>=4:
            p2D = np.array([qkp[m.queryIdx][:2] for m in matches_1NN_ratio09])
            p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_1NN_ratio09])
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D
            success_1NN, rvec_1NN, tvec_1NN, inliers_1NN = p3p_biased_RANSAC(ransac_dict,
                                                                             [1 for _ in range(len(matches_1NN_ratio09))], 10000,
                                                                             reprojection_error)
        else:
            success_1NN = False
            inliers_1NN = []
            rvec_1NN = []

        if len(matches_kNN_ratio09)>=4:
            p2D = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_ratio09])
            p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_ratio09])
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D

            success_kNN, rvec_kNN, tvec_kNN, inliers_kNN = p3p_biased_RANSAC(ransac_dict,
                                                                             [1 for _ in range(len(matches_kNN_ratio09))],
                                                                             10000,
                                                                             reprojection_error)

        else:
            success_kNN = False
            inliers_kNN = []
            rvec_kNN = []


        if len(matches_SSMC)>=4:
            p2D = np.array([qkp[m.queryIdx][:2] for m in matches_SSMC])
            p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_SSMC])
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D
            success_SSMC, rvec_SSMC, tvec_SSMC, inliers_SSMC = p3p_biased_RANSAC(ransac_dict,
                                                                                             [1 for _ in
                                                                                              range(
                                                                                                  len(matches_SSMC))],
                                                                                             10000,
                                                                                             reprojection_error)

        if len(matches_kNN_semanticratio)>=4:
            p2D = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_semanticratio])
            p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_semanticratio])
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D
            success_kNN_sem, rvec_kNN_sem, tvec_kNN_sem, inliers_kNN_sem = p3p_biased_RANSAC(ransac_dict,
                                                                                             [1 for _ in
                                                                                                                          range(
                                                                                                              len(matches_kNN_semanticratio))],
                                                                                             10000,
                                                                                             reprojection_error)

        else:
            success_kNN_sem = False
            inliers_kNN_sem = []
            rvec_kNN_sem = []

        # weighted
        p2D = np.array([qkp[m.queryIdx][:2] for m in matches_GSMC])
        p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_GSMC])

        if p2D.shape[0]>=4:
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D

            success_GSMC, rvec_GSMC, tvec_GSMC, inliers_GSMC = p3p_biased_RANSAC(ransac_dict,
                                                                                 filtered_scores, 10000,
                                                                                 reprojection_error)
        else:
            success_GSMC = False
            inliers_GSMC = [0]
            rvec_GSMC = []

        p2D = np.array([qkp[m.queryIdx][:2] for m in matches_kNN_semanticratio])
        p3D = np.array([full3Dpoints[db_p3D_ids[m.trainIdx]].xyz for m in matches_kNN_semanticratio])

        if p2D.shape[0] >= 4:
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D

            success_kNN_sem_biased, rvec_kNN_sem_biased, tvec_kNN_sem_biased, inliers_kNN_sem_biased = p3p_biased_RANSAC(ransac_dict,
                                                                                                                         all_ratios, 10000, reprojection_error)
        else:
            success_kNN_sem_biased = False
            inliers_kNN_sem_biased = [0]
            rvec_kNN_sem_biased = []


        if p2D.shape[0] >= 4:
            ransac_dict['pts_3d'] = p3D
            ransac_dict['pts_2d'] = p2D

            success_sem_biased2, rvec_sem_biased2, tvec_sem_biased2, inliers_sem_biased2 = p3p_robust_biased_sampling_and_consensus(ransac_dict,
                                                                                                         all_ratios, 10000, reprojection_error)
        else:
            success_sem_biased2 = False
            inliers_sem_biased2 = [0]
            rvec_sem_biased2 = []

        # endregion

        # save success, position error, rotation error, number of inliers, number of inliers that are real inliers?
        if success_1NN or len(rvec_1NN)>0:
            R_pred_1NN = cv.Rodrigues(rvec_1NN)[0]
            c_pred_1NN = - np.matmul(R_pred_1NN.T, tvec_1NN).flatten()
            position_error_1NN, rotation_error_1NN = compute_pose_errors(R_gt, R_pred_1NN, c_gt, c_pred_1NN)
        else:
            position_error_1NN = -1
            rotation_error_1NN = -1
            inliers_1NN = []

        if success_kNN or len(rvec_kNN)>0:
            R_pred_kNN = cv.Rodrigues(rvec_kNN)[0]
            c_pred_kNN = - np.matmul(R_pred_kNN.T, tvec_kNN).flatten()
            position_error_kNN, rotation_error_kNN = compute_pose_errors(R_gt, R_pred_kNN, c_gt, c_pred_kNN)
        else:
            position_error_kNN = -1
            rotation_error_kNN = -1
            inliers_kNN = []



        if success_SSMC or len(rvec_SSMC)>0:
            R_pred_SSMC = cv.Rodrigues(rvec_SSMC)[0]
            c_pred_SSMC =  - np.matmul(R_pred_SSMC.T, tvec_SSMC).flatten()
            position_error_SSMC, rotation_error_SSMC = compute_pose_errors(R_gt, R_pred_SSMC, c_gt,
                                                                                 c_pred_SSMC)
        else:
            position_error_SSMC = -1
            rotation_error_SSMC = -1
            inliers_SSMC = []

        if success_kNN_sem or len(rvec_kNN_sem)>0:
            R_pred_kNN_sem = cv.Rodrigues(rvec_kNN_sem)[0]
            c_pred_kNN_sem = - np.matmul(R_pred_kNN_sem.T, tvec_kNN_sem).flatten()
            position_error_kNN_sem, rotation_error_kNN_sem = compute_pose_errors(R_gt, R_pred_kNN_sem, c_gt,
                                                                                 c_pred_kNN_sem)
        else:
            position_error_kNN_sem = -1
            rotation_error_kNN_sem = -1
            inliers_kNN_sem = []

        if success_GSMC or len(rvec_GSMC)>0:
            R_pred_1NN_biased = cv.Rodrigues(rvec_GSMC)[0]
            c_pred_1NN_biased = - np.matmul(R_pred_1NN_biased.T, tvec_GSMC).flatten()
            position_error_1NN_biased, rotation_error_1NN_biased = compute_pose_errors(R_gt, R_pred_1NN_biased, c_gt,
                                                                                       c_pred_1NN_biased)
        else:
            position_error_1NN_biased = -1
            rotation_error_1NN_biased = -1

        if success_kNN_sem_biased or len(rvec_kNN_sem_biased)>0:
            R_pred_sem_biased = cv.Rodrigues(rvec_kNN_sem_biased)[0]
            c_pred_sem_biased = - np.matmul(R_pred_sem_biased.T, tvec_kNN_sem_biased).flatten()
            position_error_sem_biased, rotation_error_sem_biased = compute_pose_errors(R_gt, R_pred_sem_biased, c_gt,
                                                                                       c_pred_sem_biased)
        else:
            position_error_sem_biased = -1
            rotation_error_sem_biased = -1

        if success_sem_biased2 or len(rvec_sem_biased2)>0:
            R_pred_sem_biased2 = cv.Rodrigues(rvec_sem_biased2)[0]
            c_pred_sem_biased2 = - np.matmul(R_pred_sem_biased2.T, tvec_sem_biased2).flatten()
            position_error_sem_biased2, rotation_error_sem_biased2 = compute_pose_errors(R_gt, R_pred_sem_biased2, c_gt,c_pred_sem_biased2)
        else:
            position_error_sem_biased2 = -1
            rotation_error_sem_biased2 = -1


        query_stats = pd.DataFrame()
        #['img_name',  'experiment_type', 'k', 'ratio_test',  'matches_in', 'success', 'inliers', 'position_error', 'orientation_error']
        query_stats['img_name'] = [query_name for i in range(7)]
        query_stats['experiment_type'] = ['1NN', 'kNN', 'kNN_sem', 'SSMC', 'GSMC', 'sem_biased', 'sem_biased_sampling_and_consensus']
        query_stats['k'] = [1, k, k, 1, 1, k, k]
        query_stats['ratio_test'] = [0.9, 0.9, 1, 1, 0.9, 1, 1]
        query_stats['matches_in'] = [ len(matches_1NN_ratio09), len(matches_kNN_ratio09), len(matches_kNN_semanticratio), len(matches_SSMC), len(matches_GSMC), len(matches_kNN_semanticratio), len(matches_kNN_semanticratio)]
        query_stats['matches_in_effective'] = [len(matches_1NN_ratio09), len(matches_kNN_ratio09), len([m for m in all_ratios if m>ratio_threshold]), len(matches_SSMC), len([m for m in filtered_scores if m>0]), len([m for m in all_ratios if m>ratio_threshold]), len([m for m in all_ratios if m>ratio_threshold])]
        query_stats['success'] = [success_1NN, success_kNN, success_kNN_sem, success_SSMC, success_GSMC, success_kNN_sem_biased, success_sem_biased2]
        query_stats['inliers'] = [sum(inliers_1NN),  sum(inliers_kNN), sum(inliers_kNN_sem), sum(inliers_SSMC), sum(inliers_GSMC), sum(inliers_kNN_sem_biased), sum(inliers_sem_biased2)]
        query_stats['position_error'] = [position_error_1NN, position_error_kNN, position_error_kNN_sem, position_error_SSMC,position_error_1NN_biased, position_error_sem_biased, position_error_sem_biased2]
        query_stats['orientation_error'] = [rotation_error_1NN, rotation_error_kNN, rotation_error_kNN_sem, rotation_error_SSMC, rotation_error_1NN_biased, rotation_error_sem_biased, rotation_error_sem_biased2]

        stats_df = pd.concat([stats_df, query_stats], axis=0 )

    return stats_df


if __name__ == '__main__':
    # Add your repo path
    my_repo_path = 'ADD YOUR REPO PATH'
    slicepath = my_repo_path + '/data/Extended-CMU-Seasons/slice22'
    slice = '22'

    k = 4

    with open(os.path.join(slicepath, 'traversals.json'), 'r') as f:
        traversals_dict = json.load(f)

    # traversal_counts = list(map(len, traversals_dict.values()))
    query_names = np.array([])
    for date, img_names in traversals_dict.items():
        if date > '2010-11-02' and date < '2011-07-28':
            query_names = np.append(query_names,  np.array(img_names)[np.random.choice(len(img_names), int(np.floor(len(img_names)*0.08)), replace=False)])

    query_names = np.array(query_names)

    with open(os.path.join( my_repo_path, 'pose_estimation/experiments/repeated_structures_queries_2022_08_22_18_23_57_c0_slice22_70imgs.txt' ), 'r') as f:
        query_names = [n[:-1] for n in f.readlines()]

    now = str(datetime.datetime.now()).replace('-', '_').replace(':', '_').replace(' ', '_')[:-7]
    stats_dirname = f'exp_k{k}_n_queries{len(query_names)}_date{now}/'
    stats_dirname = os.path.join(my_repo_path, 'pose_estimation/results/pose_est_comparisons',
                                 stats_dirname)

    if not os.path.exists(stats_dirname):
        os.mkdir(stats_dirname)

    get_stats(query_names, k, slicepath, slice, stats_dirname, largest_score=True)

