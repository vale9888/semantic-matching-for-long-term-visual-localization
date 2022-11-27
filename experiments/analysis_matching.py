import sqlite3

import cv2 as cv
import numpy as np

np.random.seed(2250)
import os
import pandas as pd
import datetime

import context
from ground_truth_correspondences.get_gt_matches import get_gt_matches
from pose_estimation.utils.data_loading import get_reference_images_info_binary, load_data
from GSMC.gsmc_utils import get_point_cloud_info
from GSMC.gsmc import GSMC_score
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary
from pose_estimation.utils.matching import k_ratio_test


def get_stats(query_names, k_max, slicepath, slice, stats_dirname, gt_threshold=np.e**(-5), score_threshold=20, ratio_threshold=0.2, min_point_count=30, save_results=True, covisibility_threshold=0.2):
    '''Perform evaluation of matching for several strategies with k = 1, ..., k_max'''

    stats_df = pd.DataFrame([], columns=['img_name','k', 'tot_positives' ,'recall_or_precision','KNN_no_ratio', 'KNN_ratio_09', 'KNN_ratio_08', 'KNN_ratio_07', 'score_largest', 'score_all', 'ratio_largest', 'ratio_all', 'ratio_cutoff_largest', 'ssmc', 'ratio_covisibility'])

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

    for i, query_name in enumerate(query_names):
        print("Doing image %d / %d" %(i, len(query_names)))
        # region (0) Get ground truth matrix
        gt_match_mat = get_gt_matches(query_name, slicepath, slice)
        tot_positives = np.sum(gt_match_mat > gt_threshold)
        # endregion

        # region (1) Load data
        data_dict = load_data(query_name, slicepath, slice, load_database=False)
        camera_matrix = data_dict['camera_matrix']
        dist_coefs = data_dict['dist_coefs']
        kp_priority = data_dict['kp_priority']
        qkp = data_dict['qkp']
        q_descriptors = data_dict['qdesc']
        query_mask = data_dict['query_mask']
        c_gt = data_dict['c_gt']
        g_direction = data_dict['g_direction']
        z0 = c_gt[2]
        # endregion

        flann_matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        kplus1_nearest_matches = flann_matcher.knnMatch(q_descriptors.astype(np.float32), db_descriptors.astype(np.float32),  k=k_max+1)

        # region (2) KNN matrices

        img_stats = pd.DataFrame()
        img_stats['img_name'] = [query_name for i in range(2*k_max)]
        img_stats['k'] = [i for i in range(1, k_max+1)] + [i for i in range(1, k_max+1)]
        img_stats['tot_positives'] = [tot_positives for i in range(2 * k_max)]
        img_stats['recall_or_precision'] = ['recall' for i in range(1, k_max+1)] + ['precision' for i in range(1, k_max+1)]

        KNN_no_ratio_recalls = []
        KNN_no_ratio_precisions = []
        KNN_ratio_09_recalls = []
        KNN_ratio_09_precisions = []
        # KNN_ratio_08_recalls = []
        # KNN_ratio_08_precisions = []
        KNN_ratio_07_recalls = []
        KNN_ratio_07_precisions = []

        KNN_no_ratio_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        for k in range(1, k_max+1):
            print("Computing KNN scores for k = ", k)
            for mm in kplus1_nearest_matches:
                m = mm[k-1]
                KNN_no_ratio_df.loc[m.queryIdx, db_p3D_ids[m.trainIdx]] = 1

            KNN_no_ratio_recalls.append(np.sum(np.logical_and(gt_match_mat>gt_threshold, KNN_no_ratio_df.to_numpy()))/tot_positives)
            KNN_no_ratio_precisions.append(np.sum(np.logical_and(gt_match_mat>gt_threshold, KNN_no_ratio_df.to_numpy()))/np.sum(KNN_no_ratio_df.to_numpy()))

            KNN_ratio_09_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))),columns=full3Dpoints.keys())
            good_matches = k_ratio_test([m[:k+1] for m in kplus1_nearest_matches], 0.9)
            for mm in good_matches:
                for m in mm:
                    KNN_ratio_09_df.loc[m.queryIdx, db_p3D_ids[m.trainIdx]] = 1

            KNN_ratio_09_recalls.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_09_df.to_numpy())) / tot_positives)
            KNN_ratio_09_precisions.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_09_df.to_numpy())) / np.sum(
                    KNN_ratio_09_df.to_numpy()))

            del KNN_ratio_09_df

            # KNN_ratio_08_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
            # good_matches = k_ratio_test([m[:k + 1] for m in kplus1_nearest_matches], 0.8)
            # for mm in good_matches:
            #     for m in mm:
            #         KNN_ratio_08_df.loc[m.queryIdx, db_p3D_ids[m.trainIdx]] = 1
            #
            # KNN_ratio_08_recalls.append(
            #     np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_08_df.to_numpy())) / tot_positives)
            # KNN_ratio_08_precisions.append(
            #     np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_08_df.to_numpy())) / np.sum(
            #         KNN_ratio_08_df.to_numpy()))
            #
            # del KNN_ratio_08_df

            KNN_ratio_07_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))),columns=full3Dpoints.keys())
            good_matches = k_ratio_test([m[:k+1] for m in kplus1_nearest_matches], 0.7)
            for mm in good_matches:
                for m in mm:
                    KNN_ratio_07_df.loc[m.queryIdx, db_p3D_ids[m.trainIdx]] = 1

            KNN_ratio_07_recalls.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_07_df.to_numpy())) / tot_positives)
            KNN_ratio_07_precisions.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, KNN_ratio_07_df.to_numpy())) / np.sum(
                    KNN_ratio_07_df.to_numpy()))

            del KNN_ratio_07_df

        del KNN_no_ratio_df

        img_stats['KNN_no_ratio'] = KNN_no_ratio_recalls + KNN_no_ratio_precisions
        img_stats['KNN_ratio_09'] = KNN_ratio_09_recalls + KNN_ratio_09_precisions
        # img_stats['KNN_ratio_08'] = KNN_ratio_08_recalls + KNN_ratio_08_precisions
        img_stats['KNN_ratio_08'] = [np.NaN for _ in range(2*len(KNN_no_ratio_recalls))]
        img_stats['KNN_ratio_07'] = KNN_ratio_07_recalls + KNN_ratio_07_precisions

        # endregion

        # region (3) Scores and ratio

        # Compute and store scores
        all_scores_modified = []
        all_point_counts_modified = []
        gsmc_scores = 0
        semantic_matches = []
        semantic_neighbor = []

        print('Computing scores')
        for qkp_idx in kp_priority:

            m = kplus1_nearest_matches[qkp_idx][:-1]

            scores = []
            scores_modified = []
            point_counts = []
            point_counts_modified = []

            for neighbor_idx, putative_match in enumerate(m):
                # if query_mask[round(q_des[qkp_idx][1])][round(q_des[qkp_idx][0])] == point_cloud_info[db_p3D_ids[
                #                                                                    putative_match.trainIdx]].semantic_label:
                # if 5 < qkp[qkp_idx][0] < 1019 and 5 < qkp[qkp_idx][1] < 763 and full3Dpoints[
                #     db_p3D_ids[putative_match.trainIdx]].semantic_label in query_mask[round(qkp[qkp_idx][1]) - 5:round(
                #     qkp[qkp_idx][1]) + 5, round(qkp[qkp_idx][0]) - 5:round(qkp[qkp_idx][0]) + 5]:
                if 15 < qkp[qkp_idx][0] < 1009 and 15 < qkp[qkp_idx][1] < 753 and full3Dpoints[
                    db_p3D_ids[putative_match.trainIdx]].semantic_label in query_mask[round(qkp[qkp_idx][1]) - 15:round(
                    qkp[qkp_idx][1]) + 15, round(qkp[qkp_idx][0]) - 15:round(qkp[qkp_idx][0]) + 15] or ((0 < qkp[qkp_idx][0] < 15 or 1009 < qkp[qkp_idx][0] < 1024) and (0 < qkp[qkp_idx][1] < 15 or 753 < qkp[qkp_idx][1] < 768)):

                    if full3Dpoints[
                    db_p3D_ids[putative_match.trainIdx]].semantic_label == query_mask[round(qkp[qkp_idx][1]) , round(qkp[qkp_idx][0]) ]:
                        semantic_matches.append(putative_match)
                        semantic_neighbor.append(neighbor_idx+1)

                    # original score by Toft et al.
                    # score, _, _, totPointCounts, _ = GSMC_score(qkp[qkp_idx], db_p3D_ids[putative_match.trainIdx],
                    #                                             full3Dpoints, g_direction, camera_matrix, dist_coefs,
                    #                                             query_mask, slicepath, z0=z0, all_pids=all_pids, all_p3D=all_p3D)

                    # faster score
                    score_mod, _, _, totPointCounts_mod, _ = GSMC_score(qkp[qkp_idx], db_p3D_ids[putative_match.trainIdx],
                                                                full3Dpoints, g_direction, camera_matrix, dist_coefs,
                                                                query_mask, slicepath, z0=z0, use_covisibility=True, img_data=img_data, all_pids=all_pids, all_p3D=all_p3D)


                    # scores.append(score)
                    # point_counts.append(totPointCounts)

                    scores_modified.append(score_mod)
                    point_counts_modified.append(totPointCounts_mod)

                    gsmc_scores += 1
                    if gsmc_scores % 100 == 0:
                        print("Computed %d scores " % (gsmc_scores))

                else:
                    scores.append(0)
                    point_counts.append(0)
                    scores_modified.append(0)
                    point_counts_modified.append(0)

            # all_scores.append(scores)
            # all_point_counts.append(point_counts)
            all_scores_modified.append(scores_modified)
            all_point_counts_modified.append(point_counts_modified)

        ssmc_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        ssmc_recalls = []
        ssmc_precisions = []

        for k in range(1, k_max+1):
            for m, neighbor_idx in zip(semantic_matches, semantic_neighbor):
                if neighbor_idx == k:
                    ssmc_df.loc[m.queryIdx, db_p3D_ids[m.trainIdx]] = 1
            ssmc_recalls.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ssmc_df.to_numpy())) / tot_positives)
            ssmc_precisions.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ssmc_df.to_numpy())) / np.sum(
                    ssmc_df.to_numpy()))

        del ssmc_df

        score_all_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        ratio_all_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        ratio_cutoff_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        score_all_recalls = []
        score_all_precisions = []
        score_largest_recalls = []
        score_largest_precisions = []
        ratio_all_recalls = []
        ratio_all_precisions = []
        ratio_largest_recalls = []
        ratio_largest_precisions = []
        ratio_cutoff_recalls = []
        ratio_cutoff_precisions = []

        for k in range(1, k_max+1):
            for kp_num, (scores_list, point_count_list) in enumerate(zip(all_scores, all_point_counts)):
                for c, (sc, pc) in enumerate(zip(scores_list[:k], point_count_list[:k])):
                    qkp_idx = kp_priority[kp_num]
                    match = kplus1_nearest_matches[qkp_idx][c]
                    score_all_df.loc[qkp_idx, db_p3D_ids[match.trainIdx]] = sc
                    ratio_all_df.loc[qkp_idx, db_p3D_ids[match.trainIdx]] = sc/pc if pc > 0 else 0
                    ratio_cutoff_df.loc[qkp_idx, db_p3D_ids[match.trainIdx]] = sc/pc if pc > min_point_count else 0
            score_all_recalls.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, score_all_df.to_numpy() > score_threshold))/tot_positives)
            score_all_precisions.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, score_all_df.to_numpy() > score_threshold))/np.sum(score_all_df.to_numpy() > score_threshold))
            ratio_all_recalls.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_all_df.to_numpy() > ratio_threshold))/tot_positives)
            ratio_all_precisions.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_all_df.to_numpy() > ratio_threshold))/np.sum(ratio_all_df.to_numpy() > ratio_threshold))

            score_largest_mask = np.logical_and(score_all_df.to_numpy() >= np.outer(np.max(score_all_df.to_numpy(), axis=1), np.ones((1, score_all_df.to_numpy().shape[1]))), score_all_df.to_numpy()>score_threshold)
            score_largest_recalls.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, score_largest_mask))/tot_positives)
            score_largest_precisions.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, score_largest_mask))/np.sum(score_largest_mask))
            ratio_largest_mask = np.logical_and(ratio_all_df.to_numpy() >= np.outer(np.max(ratio_all_df.to_numpy(), axis=1), np.ones((1, ratio_all_df.to_numpy().shape[1]))), ratio_all_df.to_numpy()>ratio_threshold)
            ratio_largest_recalls.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_largest_mask))/tot_positives)
            ratio_largest_precisions.append(np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_largest_mask))/np.sum(ratio_largest_mask))

            ratio_cutoff_largest_mask = np.logical_and(ratio_cutoff_df.to_numpy() >= np.outer(np.max(ratio_cutoff_df.to_numpy(), axis=1), np.ones((1, ratio_cutoff_df.to_numpy().shape[1]))), ratio_cutoff_df.to_numpy()>ratio_threshold)
            ratio_cutoff_recalls.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_cutoff_largest_mask)) / tot_positives)
            ratio_cutoff_precisions.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_cutoff_largest_mask)) / np.sum(ratio_cutoff_largest_mask))

        del score_all_df
        del ratio_all_df
        del ratio_cutoff_df

        ratio_cutoff_covisibility_df = pd.DataFrame(np.zeros((len(qkp), len(full3Dpoints.keys()))), columns=full3Dpoints.keys())
        ratio_cutoff_covisibility_recalls = []
        ratio_cutoff_covisibility_precisions = []

        for k in range(1, k_max+1):
            for kp_num, (scores_list, point_count_list) in enumerate(zip(all_scores_modified, all_point_counts_modified)):
                for c, (sc, pc) in enumerate(zip(scores_list[:k], point_count_list[:k])):
                    qkp_idx = kp_priority[kp_num]
                    match = kplus1_nearest_matches[qkp_idx][c]
                    ratio_cutoff_covisibility_df.loc[qkp_idx, db_p3D_ids[match.trainIdx]] = sc/pc if pc > min_point_count else 0
            ratio_cutoff_covisibility_largest_mask = np.logical_and(ratio_cutoff_covisibility_df.to_numpy() >= np.outer(np.max(ratio_cutoff_covisibility_df.to_numpy(), axis=1), np.ones((1, ratio_cutoff_covisibility_df.to_numpy().shape[1]))), ratio_cutoff_covisibility_df.to_numpy()>covisibility_threshold)
            ratio_cutoff_covisibility_recalls.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_cutoff_covisibility_largest_mask)) / tot_positives)
            ratio_cutoff_covisibility_precisions.append(
                np.sum(np.logical_and(gt_match_mat > gt_threshold, ratio_cutoff_covisibility_largest_mask)) / np.sum(ratio_cutoff_covisibility_largest_mask))

        del ratio_cutoff_covisibility_df

        img_stats['score_all'] = score_all_recalls + score_all_precisions
        img_stats['ratio_all'] = ratio_all_recalls + ratio_all_precisions
        img_stats['score_largest'] = score_largest_recalls + score_largest_precisions
        img_stats['ratio_largest'] = ratio_largest_recalls + ratio_largest_precisions
        img_stats['ratio_cutoff_largest'] = ratio_cutoff_recalls + ratio_cutoff_precisions
        img_stats['ssmc'] = ssmc_recalls + ssmc_precisions
        img_stats['ratio_covisibility'] = ratio_cutoff_covisibility_recalls + ratio_cutoff_covisibility_precisions
        # endregion

        stats_df = pd.concat([stats_df, img_stats], axis=0)

    if save_results:
        stats_df.to_csv(os.path.join(stats_dirname, 'precision_recall_data.csv'))

    else:
        print(stats_df)
    return stats_df


if __name__ == '__main__':

    # Example: exp on slice 22

    # Add your repo path
    my_repo_path = 'ADD YOUR REPO PATH'
    slicepath = my_repo_path + '/data/Extended-CMU-Seasons/slice22'
    slice = '22'

    # Choose which queries to evaluate (all queries not recommended due to computational overhead)
    # Option 1 : predefined names
    # query_names = [query_name, 'img_05873_c0_1287504339409812us.jpg', 'img_07390_c1_1290445321344740us.jpg',
    #                'img_07503_c1_1284563907121851us.jpg', 'img_07843_c0_1292959620331676us.jpg'] (these were used in the experiment exp_kmax8_n_queries5_date2022_07_21_09_01_23)  # [query_name]

    # Option 2 : 150 random queries (stratified sampling by traversal date)
    # query_names = [n for r, s, f in os.walk(os.path.join(slicepath, 'query')) for n in f if n.endswith('.jpg')]
    # query_names = np.array(query_names)[np.random.choice(len(query_names), 150, replace=False)]

    # Option 3 : load pics with repeated structures
    with open(my_repo_path + '/experiments/repeated_structures_queries_c0_slice22_110imgs.txt', 'r') as f:
        query_names = [n[:-1] for n in f.readlines()]

    # Set number of neighbors to explore - change this
    k_max = 6

    # Setup the save folder for the experiment
    now = str(datetime.datetime.now()).replace('-', '_').replace(':', '_').replace(' ', '_')[:-7]
    stats_dirname = f'exp_kmax{k_max}_n_queries{len(query_names)}_date{now}/'
    stats_dirname = os.path.join(my_repo_path + '/experiments/results/matching_comparisons', stats_dirname)
    if not os.path.exists(stats_dirname):
        os.mkdir(stats_dirname)
    stats_df = get_stats(query_names, k_max,  slicepath, slice, stats_dirname, ratio_threshold=0.001)



