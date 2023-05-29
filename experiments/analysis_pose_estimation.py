import json
import sqlite3

import cv2 as cv
import numpy as np

np.random.seed(2250)
import os
import pandas as pd
import datetime

import time
import sys

sys.path.append( os.path.dirname( os.path.dirname( os.path.realpath( __file__ ) ) ) )

from pose_estimation.utils.matching import ratio_test
from pose_estimation.utils.data_loading import get_ref_2D_data, get_point_cloud_info
from GSMC.gsmc import GSMC_score, compute_gsmc_score_torch
from experiments.analysis_matching import load_data, k_ratio_test
from pose_estimation.utils.eval import compute_pose_errors
from pose_estimation.RANSAC_custom import p3p_biased_RANSAC, p3p_robust_biased_sampling_and_consensus
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary

rootname = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

width = 1024
height = 768

#


    # Action items
    # 0) Check rot_mat (errori compilazione e semantica) ok
    # 1) Fix stellina ok
    # 2) Fix distorsioni ok
    # 3) Check semantica distorsioni (previously ok)
    # 4) Fix visibilità (prima pos, poi angolo) ok
    # 5) Check visibilità


        # p2D_h_curr = p2D_h_curr / p2D_h_curr[ :, :, -1: ]
        # p2D_curr = p2D_h_curr[ :, :, :2 ]
        #
        # # p2D_h_curr.int()
        #
        # # visibility_mask = compute_visibility_mask( points_3d_coords, ctr_array, point_cloud_info )
        #
        # # p2D = np.array( p2D )
        # # p2D = np.rint( p2D ).astype( np.int32 )  # TODO: vediamo se con int32 o int64 va più veloce
        #
        # # Verify which point projections fall into the image
        # img_plane_mask = torch.logical_and( p2D_curr > np.zeros_like( p2D_curr ),
        #                                     p2D_curr < np.array( [ width, height ] )[ None, :, None ]
        #                                 ).all( axis = 1 )
        #
        # # candidate_match_mask = np.logical_and( visibility_mask, img_plane_mask )
        #
        # pc_labels = np.hstack( [ v.semantic_label for _, v in point_cloud_info.items() ] )
        # # query_mask
        #
        # candidate_match_mask = np.where( candidate_match_mask )
        # pid_mask = candidate_match_mask[ 1 ]
        # center_mask = candidate_match_mask[ 0 ]
        # pc_labels = pc_labels[ pid_mask ]
        #
        # visible_pts_coords = p2D[ center_mask, :, pid_mask ]
        # visible_pts_labels = query_mask[ visible_pts_coords[ :, 1 ], visible_pts_coords[ :, 0 ] ]
        #
        # equal_labels = (pc_labels == visible_pts_labels)
        #
        # scores = np.bincount( center_mask, weights = equal_labels ) # (bs,1)
        # best_center = np.unique( center_mask )[ np.argmax( scores ) ]



    # train_features, train_labels = next( iter( train_dataloader ) )

    # p2D_h = camera_internals @ rt_matrix @ p3D_h  # esplode!



    # Action items
    # 4) Capire come calcolare p2h
    # --> end compute_projections







    # Action items
    # 1) Ignoriamo la 4a dimensione (situazione di ieri)
    # 2) Iteriamo sulla 4a dimensione (iteriamo sui matches)
    # 3) Facciamo la proiezion con queste shapes (3,3) (360, 3, 4) (4, 02916)
    # 4) Calcoliamo il semantic score del match --> crei array di 92k prima del for, e rimpi la i-th entry con il valore
    # 5) Proviamo a parallelizzare il for:
    #    5i)  metti il body del for dentro una funzioncina
    #    5ii)     # with multiprocessing.Pool() as pool:
    #     #            for result in pool.imap( body_for, array_indici_matches ):
    #     #                print( result )
    # guarda su
    # vediamo quanto ci mettiamo
    #
    #
    # visibilità (visualizzazioni visibilità)

    # for p in :
    #




    # 92916 * 3 * 360 * 1078



    # endregion
    # p3d, p2d, ctr_array = compute_projections_new( match_pts_2d, match_pts_3d, R, z0,
    #                                                g_direction, point_cloud_info,
    #                                                camera_internals, dist_coefs, n_angles = n_angles )


# def funzioncina( kp_priority, camera_matrix, width, height, kplus1_nearest_matches, qkp, full3Dpoints, db_p3D_ids,
#                  g_direction, dist_coefs, query_mask, z0, window, ratio_threshold, largest_score ):
#     gsmc_scores = 0
#     matches_kNN_semanticratio = [ ]
#     all_ratios = [ ]
#
#     matches_GSMC = [ ]
#     filtered_scores = [ ]
#
#     matches_SSMC = [ ]
#
#     for qkp_idx in kp_priority:
#
#         m = kplus1_nearest_matches[ qkp_idx ][ :-1 ]
#         ratios = [ ]
#
#         for neighbor_idx, putative_match in enumerate( m ):
#
#             qkp_x = min( round( qkp[ qkp_idx ][ 0 ] ), width - 1 )
#             qkp_y = min( round( qkp[ qkp_idx ][ 1 ] ), height - 1 )
#
#             sem = full3Dpoints[ db_p3D_ids[ putative_match.trainIdx ] ].semantic_label
#
#             # region refactoring GSMC 3D
#             # p3D = extract_point_cloud_coords( full3Dpoints )
#             # score, n_visible_points, ctr, ctr_id = compute_GSMC_score_compiled( np.array( [ qkp_x, qkp_y ] ), db_p3D_ids[ putative_match.trainIdx ], z0,
#             #                              g_direction, camera_matrix,
#             #                              dist_coefs, query_mask, p3D, full3Dpoints )
#             # continue
#             # endregion
#
#             if neighbor_idx == 0 and query_mask[ qkp_y, qkp_x ] == sem:
#                 matches_SSMC.append( putative_match )
#
#             if (window < qkp_x < width - window and window < qkp_y < height - window and \
#                 sem in query_mask[ qkp_y - window:qkp_y + window, qkp_x - window:qkp_x + window ]) or \
#                     not (window < qkp_x < width - window and window < qkp_y < height - window):
#
#                 score, _, _, totPointCounts, _ = GSMC_score( qkp[ qkp_idx ], db_p3D_ids[ putative_match.trainIdx ],
#                                                              full3Dpoints, g_direction, camera_matrix, dist_coefs,
#                                                              query_mask, slicepath, z0 = z0 )
#
#                 gsmc_scores += 1
#                 if gsmc_scores % 100 == 0:
#                     print( "Computed %d scores " % (gsmc_scores) )
#
#                 if totPointCounts > 0 and score / totPointCounts > ratio_threshold:
#                     ratios.append( score / totPointCounts )
#                 else:
#                     ratios.append( 0 )
#
#                 if neighbor_idx == 0 and putative_match.distance < 0.9 * m[ 1 ].distance:
#                     matches_GSMC.append( putative_match )
#                     filtered_scores.append( score )
#
#
#             else:
#                 ratios.append( 0 )
#
#                 if neighbor_idx == 0 and putative_match.distance < 0.9 * m[ 1 ].distance:
#                     score, _, _, totPointCounts, _ = GSMC_score( qkp[ qkp_idx ],
#                                                                  db_p3D_ids[ putative_match.trainIdx ],
#                                                                  full3Dpoints, g_direction, camera_matrix,
#                                                                  dist_coefs, query_mask, slicepath,
#                                                                  z0 = z0 )  # , use_covisibility=True,
#                     # img_data=img_data)
#                     gsmc_scores += 1
#                     if gsmc_scores % 100 == 0:
#                         print( "Computed %d scores " % (gsmc_scores) )
#
#                     matches_GSMC.append( putative_match )
#                     filtered_scores.append( score )
#
#         # if largest_score:
#         #     top_idx = np.argmax( ratios )
#         #     all_ratios.append( ratios[ top_idx ] )
#         #     matches_kNN_semanticratio.append( m[ top_idx ] )
#         #
#         # else:
#         #     all_ratios += [ r for r in ratios if r != 0 ]
#         #     matches_kNN_semanticratio += [ match for c, match in enumerate( m ) if ratios[ c ] != 0 ]


def estimate_pose( match_pts_2d, match_pts_3d, sampling_weights, internals, dist_coefs, R_gt, c_gt,
                   reprojection_err = 8, n_iterations = 10000, inlier_weights = None ):
    '''

    :param inlier_weights:
    :param match_pts_2d:
    :param match_pts_3d:
    :param sampling_weights:
    :param reprojection_err:
    :param dist_coefs:
    :param internals:
    :return:
    '''

    assert len( match_pts_3d ) == len( match_pts_2d )

    position_err = -1
    rotation_err = -1
    n_inliers = 0
    success = False
    rvec_est = None
    tvec_est = None
    inliers = None

    if len( match_pts_3d ) >= 4:
        success, rvec_est, tvec_est, inliers = p3p_biased_RANSAC( match_pts_2d, match_pts_3d,
                                                                  sampling_weights = sampling_weights,
                                                                  num_iterations = n_iterations,
                                                                  r = reprojection_err,
                                                                  camera_matrix = internals,
                                                                  dist_coeff = dist_coefs,
                                                                  inlier_weights=inlier_weights)
    if rvec_est is not None:
        R_est = cv.Rodrigues( rvec_est )[ 0 ]
        c_est = - np.matmul( R_est.T, tvec_est ).flatten()
        position_err, rotation_err = compute_pose_errors( R_gt, R_est, c_gt, c_est )
        n_inliers = sum( inliers )

    return [ success, position_err, rotation_err, n_inliers ]


def save_exp_results( query_name, experiment_type, k, ratio_threshold, matches_in, pose_results, exp_name ):
    if not exp_name is None:
        gsheet = GSheet( worksheet_name = exp_name )
        values = [ query_name, experiment_type, str(k),
                   str(ratio_threshold), str(matches_in) ] + [str(el) for el in pose_results]
        gsheet.store_row( values )

def get_stats(query_names, k, slicepath, slice, savepath, ratio_threshold=0.2, height=768, width=1024, window=15, largest_score=False):
    stats_df = pd.DataFrame([], columns=['img_name',  'experiment_type','k', 'ratio_test', 'matches_in', 'matches_in_effective','success', 'inliers', 'position_error', 'orientation_error'])

    # Load database information only once
    database_path = slicepath + '/database' + str(slice) + '.db'
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    imagesbin_path = slicepath + '/sparse/images.bin'
    db_image_ids, db_kp_coords_x, db_kp_coords_y, db_p3D_ids, db_descriptors, db_image_names, cursor = _get_reference_images_info_binary(
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

        # region fast debugging only
        qkp = qkp[:30]
        q_descriptors = q_descriptors[:30]
        # endregion

        n_qkp = qkp.shape[ 0 ]


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

        # any additional masks for other matching strategies
        # endregion

        scores, bestCs, bestRs, totPointCounts = compute_gsmc_score_torch( match_pts_2d, match_pts_3d, full3Dpoints,
                                                                           all_p3D, z0, g_direction, camera_matrix,
                                                                           dist_coefs, query_mask,
                                                                           n_angles = 180)

        ratios = np.divide( scores, totPointCounts, out=np.zeros_like(scores),
                            where = totPointCounts > 0 )  # sets to 0 where totPointCounts is zero

        # setup our methods masks
        # 0. add two base checks: ssmc + visibilitymc
        # pad_query_mask = np.pad( query_mask, smc_window_size, 'constant', constant_values = -1 )

        match_px_2d_inf = np.clip( match_pts_2d - smc_window_size, a_min = 0, a_max = np.inf  ).astype(int)
        match_px_2d_sup = np.stack( [ np.clip( match_pts_2d[ :,0] + smc_window_size, a_min = -np.inf, a_max = height ), np.clip( match_pts_2d[:, 1] + smc_window_size, a_min = -np.inf, a_max = width )], axis = 1 ).astype(int)
        all_neighboring_labels = query_mask[ match_px_2d_inf[:, 1]:match_px_2d_sup[:, 1], match_px_2d_inf[:, 0]:match_px_2d_sup[:, 0] ]
        match_lab_3d = all_3d_sem_labels[ matches_ids[ :, 1 ] ][:,None]
        smc_mask = (match_lab_3d == all_neighboring_labels).any(axis=1)

        sm_thresh = 0.2
        match_sm_thresh_mask = (ratios > sm_thresh) & smc_mask

        best_ratios_ids = np.argmax( ratios.reshape( (-1, k) ), axis = 1 ) + k * np.arange( n_qkp )
        match_sm_best_mask = np.zeros_like( match_sm_thresh_mask )
        match_sm_best_mask[ best_ratios_ids ] = True
        # adding absolute threshold on top
        match_sm_best_mask[ ratios < sm_thresh ] = False


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

# Below some individual experiments of interest
def exp1():
    "Slice 25, 5 imgs, debugging purposes"
    slicepath = rootname + '/data/Extended-CMU-Seasons/slice25'
    slice = '25'

    k = 2

    # with open(os.path.join(slicepath, 'traversals.json'), 'r') as f:
    #     traversals_dict = json.load(f)
    #
    # # traversal_counts = list(map(len, traversals_dict.values()))
    # query_names = np.array([])
    # for date, img_names in traversals_dict.items():
    #     if date > '2010-11-02' and date < '2011-07-28':
    #         query_names = np.append(query_names,  np.array(img_names)[np.random.choice(len(img_names), int(np.floor(len(img_names)*0.08)), replace=False)])
    #
    # query_names = np.array(query_names)

    # with open(os.path.join( my_repo_path, 'experiments/repeated_structures_queries_2022_08_22_18_23_57_c0_slice22_70imgs.txt' ), 'r') as f:
    #     query_names = [n[:-1] for n in f.readlines()]

    # query_df = pd.read_csv(
    #     os.path.join( rootname, 'experiments/match_scarcity_queries_slice25_100imgs_2023_05_08_23_37.csv' ) )
    # query_names = query_df.img_name

    query_names = [ 'img_05463_c1_1311875340184523us.jpg',
                    'img_05474_c0_1311875343317817us.jpg',
                    'img_05492_c0_1311875348517892us.jpg',
                    'img_05495_c0_1311875349317851us.jpg',
                    'img_05497_c0_1311875349917905us.jpg'
                    ]

    now = str( datetime.datetime.now() ).replace( '-', '_' ).replace( ':', '_' ).replace( ' ', '_' )[ :-7 ]
    exp_name = f'exp_k{k}_n_queries{len( query_names )}_slice{slice}_date{now}/'
    # stats_dirname = os.path.join( rootname, 'pose_estimation/results/pose_est_comparisons',
    #                               stats_dirname )

    # if not os.path.exists( stats_dirname ):
    #     os.mkdir( stats_dirname )

    get_stats( query_names, k, slicepath, slice, exp_name )
    # stats.to_csv( stats_dirname )

def exp2( dry_run=True ):
    "Slice 22, same queries with huge MS as the thesis"
    slicepath = rootname + '/data/Extended-CMU-Seasons/slice22'
    slice = '22'

    k = 4


    results_df = pd.read_csv( os.path.join( rootname, 'pose_estimation/results/pose_est_comparisons/exp_k4_n_queries101_date2022_08_23_09_28_57/pose_est_stats.csv') )

    query_names = results_df.img_name.unique()[11:]

    now = str( datetime.datetime.now() ).replace( '-', '_' ).replace( ':', '_' ).replace( ' ', '_' )[ :-7 ]
    exp_name = f'exp2_k{k}_n_queries{len( query_names )}_slice{slice}_date{now}/' if not dry_run else None
    get_stats( query_names, k, slicepath, slice, exp_name )


if __name__ == '__main__':
    # exp2( dry_run = False )
    exp1()