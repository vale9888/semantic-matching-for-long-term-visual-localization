import cv2 as cv
import numpy as np
import math

import sys
import os
import time

import torch
from torch.utils.data import DataLoader


sys.path.append( os.path.dirname( os.path.realpath( __file__ ) ) )

from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary
from GSMC.gsmc_utils import get_rotation_matrix, get_rotation_matrix_nd_torch
from utils.utils import my_timeit

import matplotlib.pyplot as plt



width   =   1024
height  =   768

@my_timeit
def to_homogeneous(pts, coord_axis=-1):
    if coord_axis==0:
        return np.concatenate( [ pts, np.ones( pts.shape[ 1 ] )[ None, : ] ] , axis = 0 )
    return np.hstack( [ pts, np.ones( pts.shape[ 0 ] )[ :, None ] ] )

def to_homogeneous_torch(pts, coord_axis = -1, device = None ):

    if device is None:
        device = "cuda"

    if coord_axis == 0:
        return torch.cat( [ pts, torch.ones( ( 1, pts.shape[ 1 ]), device = device ) ], dim = 0 )
    else:
        return torch.cat( [ pts, torch.ones( (pts.shape[ 0 ], 1), device = device ) ], dim = 1 )



def project_point_cloud(rvec, tvec, camera_internals, point_cloud_info, project_point_ids, query_mask, height=768,
                        width=1024, distortion_coefs=None, p3D=None, sem_labels=None):
    '''
    Compute the geometric semantic score by projecting the point cloud through a given camera orientation
    :param rvec: camera rotation vector
    :param tvec: camera translation vector
    :param camera_internals: camera calibration matrix
    :param point_cloud_info: full point cloud object
    :param project_point_ids: ids of points to be projected
    :param query_mask: semantic mask of the query (array)
    :param height: picture height
    :param width: picture width
    :param full_result: boolean, whether to return score and additional info
    :param distortion_coefs: distortion coefficients
    :param p3D: pre-computed 3D point locations
    :param sem_labels: pre-computed point semantic labels

    :return: semantic score
    '''

    # Load point cloud and semantic labels where not provided
    if p3D is None:
        p3D = np.array([point_cloud_info[pid].xyz for pid in project_point_ids])

    if sem_labels is None:
        sem_labels = np.array([point_cloud_info[pid].semantic_label for pid in project_point_ids])

    # Project points
    p2D = cv.projectPoints(p3D, rvec, tvec, camera_internals, np.array(distortion_coefs))[0].reshape(
        (len(project_point_ids), 2))
    p2D = np.rint(p2D).astype(np.int16)

    # Verify which point projections fall into the image
    within_range = np.logical_and(np.greater(p2D, np.zeros_like(p2D)), np.greater(
        np.append(np.ones((len(p2D), 1)) * width, np.ones((len(p2D), 1)) * height, axis=1), p2D)).all(axis=1)

    # Compute score
    score = np.equal(sem_labels[within_range], query_mask[p2D[within_range, 1], p2D[within_range, 0]]).sum()

    return score


def verify_visibility(C, point_info):

    difference = C - point_info.xyz
    difference_norm = math.sqrt(sum(difference ** 2))
    cos_angle = difference.dot(point_info.v) / difference_norm
    cmp_angle = point_info.theta / 2

    if np.arccos(cos_angle) < cmp_angle:
        return True
    return False


def GSMC_score(p2D, p3D_id, point_cloud_info, g_direction, camera_internals, dist_coefs, query_mask, slicepath, z0=None,
               z0_dict=None, exact_filtering=True, use_covisibility=False, img_data=None):
    """
    Computation of the Geometric Semantic Match Consistency score for the match p2D-p3D by projecting the visible points onto the image plane.
    We implement two versions: one that is closely based on the work of [Toft '17], and one which does not re-compute visible points at every pose, but rather chooses based on covisibility with the considered match.

    p2D: the coordinates of the image point in the image reference system
    p3D_id: the unique id of the matched point
    point_cloud_info: list of FullPoints3D objects with all necessary information
    g_direction: direction of g in camera coordinates
    z0: height of the camera
    camera_internals: matrix of camera params
    dist_coefs: 4 coefficients for radial and tangential distortion, [k1, k2, p1, p2]
    query_mask: semantic mask of the query
    slicepath: location of the currently analysed slice
    z0: camera height
    use_covisibility=False: whether to pick points to be projected based on covisibility
    img_data=None: reference images data for the covisibility check - pass to avoid re-loading at every match iteration
    all_pids=None: array containing all point ids - pass to avoid re-loading at every match iteration
    all_p3D=None: array containing all point coordinates - pass to avoid re-loading at every match iteration
    """

    # region Refactoring numba
    match_2d_coords = p2Dj
    match_3d_id = p3D_id
    points_3d_coords = extract_point_cloud_coords(point_cloud_info)

    compute_GSMC_score_compiled( match_2d_coords, match_3d_id, z0, g_direction, camera_internals, dist_coefs,
                                 query_mask, points_3d_coords, point_cloud_info, n_angles = 360 )
    # endregion


    bestScore = 0
    bestC = []
    tot_points_count = 0
    bestR = []
    bestPointIds = []

    x_dir = np.linalg.inv(camera_internals).dot(np.array([p2D[0], p2D[1], 1]))

    alpha = np.arccos(np.array(g_direction).dot(x_dir)) - np.pi / 2
    X_coords = np.array(point_cloud_info[p3D_id].xyz)
    if z0 is None:
        z0 = z0_dict[str(p3D_id)]
    R = np.abs(X_coords[2] - z0) / np.abs(np.tan(alpha))
    C = np.array([X_coords[0] + R, X_coords[1], z0])

    if not ((point_cloud_info[p3D_id].dlow * 0.9) ** 2 < np.sum((C - X_coords) ** 2) < (
            point_cloud_info[p3D_id].dup * 1.1) ** 2):
        return 0, [], [], 0, []

    if use_covisibility:
        if img_data == None:
            images_path = slicepath + '/sparse/images.bin'
            img_data = read_images_binary(images_path)
        seen_img_ids = point_cloud_info[p3D_id].image_ids
        viewing_direction = point_cloud_info[p3D_id].v
        viewing_angle = point_cloud_info[
                            p3D_id].theta / 2  # hemiangle, as the whole thing is from an extrema to the other

        visibility_info_pids = set()

        for iid in seen_img_ids:
            visibility_info_pids = visibility_info_pids.union([pt for pt in img_data[iid].point3D_ids if pt != -1])

        project_points_ids = list(visibility_info_pids)
        p3D = np.array([point_cloud_info[pid].xyz for pid in project_points_ids])
        sem_labels = np.array([point_cloud_info[pid].semantic_label for pid in project_points_ids])

        if len(project_points_ids) == 0:
            return 0, [], [], 0, []

        tot_points_count = len(project_points_ids)

        phi = - max(min(2 * viewing_angle, np.pi), np.pi / 6)
        ref_phi = np.arctan(viewing_direction[1] / viewing_direction[0])

        while phi < max(min(2 * viewing_angle, np.pi), np.pi / 6):
            C = np.array([X_coords[0] + R * np.cos(ref_phi + phi), X_coords[1] + R * np.sin(ref_phi + phi), z0])
            rotmat = get_rotation_matrix(g_direction, x_dir, C, X_coords)
            tvec = - rotmat.dot(C)
            rvec = cv.Rodrigues(rotmat)[0]

            score = project_point_cloud(rvec, tvec, camera_internals, point_cloud_info,
                                        project_points_ids, query_mask,
                                        distortion_coefs=dist_coefs,
                                        p3D=p3D, sem_labels=sem_labels)

            if score > bestScore:
                bestScore = score
                bestC = C
                bestR = rotmat

                bestPointIds = project_points_ids

            phi += 1 / 180 * np.pi


    else:

        intermediate_project_points_ids = []

        for pid, info in point_cloud_info.items():
            if (info.xyz[0] - X_coords[0]) ** 2 + (info.xyz[1] - X_coords[1]) ** 2 < R ** 2:
                intermediate_project_points_ids.append(pid)

        intermediate1_project_points_ids = []

        # Exact filtering cuts at the most extreme observed distances, otherwise leave some tolerance
        if exact_filtering:
            for pid in intermediate_project_points_ids:
                pt = point_cloud_info[pid]
                ssd = np.sum((C - pt.xyz) ** 2)
                if pt.dlow ** 2 <= ssd <= pt.dup ** 2:
                    intermediate1_project_points_ids.append(pid)
        else:
            for pid in intermediate_project_points_ids:
                pt = point_cloud_info[pid]
                ssd = np.sum((C - pt.xyz) ** 2)
                if (pt.dlow * 0.9) ** 2 < ssd < (pt.dup * 1.1) ** 2:
                    intermediate1_project_points_ids.append(pid)
        if len(intermediate1_project_points_ids) < 30:
            return 0, [], [], len(intermediate_project_points_ids), []

        phi = 0
        while phi < 360:
            C = np.array([X_coords[0] + R * np.cos(phi * np.pi / 180), X_coords[1] + R * np.sin(phi * np.pi / 180), z0])
            project_points_ids = []

            for pid in intermediate1_project_points_ids:

                if verify_visibility(C, point_cloud_info[pid]):
                    project_points_ids.append(pid)

            if len(project_points_ids) > 0:

                rotmat = get_rotation_matrix(g_direction, x_dir, C, X_coords)
                tvec = - rotmat.dot(C)
                rvec = cv.Rodrigues(rotmat)[0]

                score = project_point_cloud(rvec, tvec, camera_internals, point_cloud_info, project_points_ids,
                                            query_mask, distortion_coefs=dist_coefs)

                if score > bestScore:
                    bestScore = score
                    bestC = C
                    bestR = rotmat
                    tot_points_count = len(project_points_ids)
                    bestPointIds = project_points_ids

                phi += 1
            else:
                # To speed up the computation, only reduce the number of explored poses around areas with no projected points
                phi += 10

    return bestScore, bestC, bestR, tot_points_count, bestPointIds


def compute_gsmc_score_torch( match_pts_2d, match_pts_3d, point_cloud_info, z0, g_direction, camera_internals,
                              dist_coefs, query_mask, n_angles = 360, c_gt=None, R_gt=None ):

    n_matches = match_pts_2d.shape[0]  # 1078

    p3D = np.stack( [ v.xyz for _, v in point_cloud_info.items() ], axis=-1 )

    # device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    device = "cpu"
    print( f'Torch is using device: {device}' )

    # Convertiamo in torch
    match_pts_2d = torch.from_numpy(match_pts_2d).float().to(device)
    match_pts_3d = torch.from_numpy(match_pts_3d).float().to(device)
    camera_internals = torch.from_numpy(camera_internals).float().to(device)
    dist_coefs = torch.from_numpy( dist_coefs ).float().to( device )
    g_direction = torch.from_numpy( g_direction ).float().to( device )
    p3D = torch.from_numpy( p3D ).float().to( device )

    match_pts_2d_h = to_homogeneous_torch(match_pts_2d, coord_axis = -1, device = device )  # (1078, 3)
    match_pts_2d_h = match_pts_2d_h @ torch.linalg.inv( camera_internals ).T  # (1078, 3) @ (3, 3) = (1078, 3)
    match_pts_2d_h = match_pts_2d_h / torch.linalg.norm( match_pts_2d_h, dim = -1 )[:, None]

    alpha = torch.arccos( match_pts_2d_h @ g_direction ) - math.pi / 2  # (1078, 3) @ (3, 1) = (1078, 1)
    R = torch.abs(match_pts_3d[:, 2] - z0 ) /  torch.abs( torch.tan( alpha ) )

    phi = torch.arange( n_angles , device = device)

    # ctr is a 3d-matrix (n_query_matches, n_angles, 3) where the (i,j) location represent the coordinates
    # of the tentative camera center, once fixed the i-th match and the j-th angle
    ctr_x = match_pts_3d[ :, 0, None] + R[:, None] * torch.cos( phi * math.pi / 180 )[None, :]
    ctr_y = match_pts_3d[ :, 1, None] + R[:, None] * torch.sin( phi * math.pi / 180 )[None, :]
    ctr_z = torch.full( ctr_x.shape, z0 , device = device)
    ctr = torch.stack( [ ctr_x, ctr_y, ctr_z ], dim = -1 )

    rotmat_nd = get_rotation_matrix_nd_torch( g_direction, match_pts_2d_h.T,
                                              torch.movedim(ctr, (0, 1, 2), (2, 1, 0)),  # ctr.T,
                                              match_pts_3d.T ,device = device )
    # rotmat_nd = torch.swapaxes(rotmat_nd, 0, 1)

    # rotmat_nd = torch.swapaxes( get_rotation_matrix_nd( g_direction, match_pts_2d.T, ctr.T, match_pts_3d.T), 0, 1 )  # (360, 1078, 3, 3)
    # rotmat_nd = torch.ones( (1078, 360,  3, 3), device=device)

    tvec_nd = - torch.sum( rotmat_nd * ctr[ :, :, None ], dim = -1 )

    rt_matrix = torch.concatenate( [ rotmat_nd, tvec_nd[ :, :, :, None ] ], dim = -1 )

    # p3D_h = to_homogeneous_torch( p3D, coord_axis = 0, device = device )  # (4, 92916)

    rt_matrix_flat = torch.reshape(rt_matrix, (rt_matrix.shape[0] * rt_matrix.shape[1],
                                               rt_matrix.shape[2],
                                               rt_matrix.shape[3]))
    ctr_flat = torch.reshape(ctr, (rt_matrix.shape[0] * rt_matrix.shape[1],
                                    ctr.shape[2],
                                    1
                                   ) )

    tensor_dataset = torch.utils.data.TensorDataset(rt_matrix_flat, ctr_flat)

    num_semantic_inliers = torch.zeros( rt_matrix.shape[ 0 ] * rt_matrix.shape[ 1 ], device = device )  # (bs, 1)

    del ctr_x, ctr_y, ctr_z, ctr, rt_matrix, rotmat_nd, tvec_nd  # TODO: delete all useless tensor from now on
    torch.cuda.empty_cache()

    batch_size = 4100  # 2048
    rtc_dataloader = DataLoader( tensor_dataset, batch_size = batch_size, shuffle = False )

    thr_d_low, thr_d_up, mid_camera_directions, thr_cos_angle = get_visibility_thresholds_torch( point_cloud_info, device = device )

    # load semantic data
    pc_labels = torch.Tensor( [ v.semantic_label for _, v in point_cloud_info.items() ] ).to( device )
    query_mask = torch.from_numpy(query_mask).to(device)

    start_time = time.time()
    for curr_idx, (curr_rt, curr_ctr) in enumerate(rtc_dataloader):
        p2D_curr = compute_projections_torch( curr_rt, p3D, camera_internals, dist_coefs, device )
        # p2D_h_curr = camera_internals @ curr_rt @ p3D_h  # (3, 3) @ (bs, 3, 4) @ (4, 92916) = (bs, 3, 92916)

        visibility_mask = compute_visibility_mask_torch( p3D, curr_ctr, thr_d_low, thr_d_up, mid_camera_directions, thr_cos_angle, device,
                                                         curr_rt, c_gt, R_gt)


        p2D_curr = p2D_curr.int()

        img_plane_mask = torch.logical_and( p2D_curr > torch.zeros_like( p2D_curr , device = device),
                                            p2D_curr < torch.Tensor( [ width, height ] )[ None, :, None ].to(device)
                                          ).all( dim = 1 )
        # visibility_mask = torch.ones_like(img_plane_mask)

        candidate_match_mask = torch.logical_and( visibility_mask, img_plane_mask )

        # query_mask

        candidate_match_mask = torch.where( candidate_match_mask )
        pid_mask = candidate_match_mask[ 1 ]
        center_mask = candidate_match_mask[ 0 ]
        pc_labels_curr = pc_labels[ pid_mask ]

        visible_pts_coords = p2D_curr[ center_mask, :, pid_mask ]
        visible_pts_labels = query_mask[ visible_pts_coords[ :, 1 ], visible_pts_coords[ :, 0 ] ]

        equal_labels = (pc_labels_curr == visible_pts_labels)

        scores = torch.bincount( center_mask, weights = equal_labels, minlength = curr_rt.shape[0] )

        num_semantic_inliers[curr_idx * batch_size: (curr_idx + 1) * batch_size] = scores  # <-- stellina
    print( f'Duration: {time.time() - start_time}' )

    exit(0)
