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



def project_point_cloud( rvec, tvec, camera_internals, point_cloud_info, project_point_ids, query_mask, height = 768,
                         width = 1024, distortion_coefs = None, p3D = None, sem_labels = None ):
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
        p3D = np.array( [ point_cloud_info[ pid ].xyz for pid in project_point_ids ] )

    if sem_labels is None:
        sem_labels = np.array( [ point_cloud_info[ pid ].semantic_label for pid in project_point_ids ] )

    # Project points
    p2D = cv.projectPoints( p3D, rvec, tvec, camera_internals, np.array( distortion_coefs ) )[ 0 ].reshape(
        (len( project_point_ids ), 2) )
    p2D = np.rint( p2D ).astype( np.int16 )

    # Verify which point projections fall into the image
    within_range = np.logical_and( np.greater( p2D, np.zeros_like( p2D ) ), np.greater(
        np.append( np.ones( (len( p2D ), 1) ) * width, np.ones( (len( p2D ), 1) ) * height, axis = 1 ), p2D ) ).all(
        axis = 1 )

    # Compute score
    score = np.equal( sem_labels[ within_range ], query_mask[ p2D[ within_range, 1 ], p2D[ within_range, 0 ] ] ).sum()

    return score

@my_timeit
def compute_visibility_mask( p3D, ctr_array, point_cloud_info ):
    # dist_matrix = np.array( [ np.linalg.norm( p3D - ctr, axis = 1 ) for ctr in ctr_array ] )  # dim: angles x n_pts  # for loop, dist
    # diff_matrix = np.array( [ p3D - ctr for ctr in ctr_array ] )  # dim: angles x n_pts x 3  # for loop, diff

    diff_matrix = p3D[ None, ] - ctr_array[ :, None, ]  # dim: angles x n_pts x 3  # broadcasting
    dist_matrix = np.linalg.norm( diff_matrix, axis = -1 )  # dim: angles x n_pts

    thr_low = np.hstack( [ v.dlow for _, v in point_cloud_info.items() ] )
    thr_up = np.hstack( [ v.dup for _, v in point_cloud_info.items() ] )
    dist_matrix_mask = np.logical_and( dist_matrix > thr_low, dist_matrix < thr_up )  # n_angle x n_pts

    mid_camera_directions = np.vstack( [ v.v for _, v in point_cloud_info.items() ] )  # n_pts x 3
    thr_angle = np.hstack( [ v.theta * 2 for _, v in point_cloud_info.items() ] )  # n_pts
    thr_cos_angle = np.cos( thr_angle )  # n_pts

    match_directions = diff_matrix / dist_matrix[ :, :, None ]
    cos_angle = np.sum( match_directions * mid_camera_directions, axis = -1 )  # cos_similarity
    angle_mask = cos_angle > thr_cos_angle

    visibility_mask = np.logical_and( dist_matrix_mask, angle_mask )

    # return visibility_mask  # TODO: check angle filtering
    return dist_matrix_mask

def get_visibility_thresholds_torch( point_cloud_info, device = "cuda" ):
    thr_d_low = torch.Tensor( [ v.dlow for _, v in point_cloud_info.items() ] ).to( device )
    thr_d_up = torch.Tensor( [ v.dup for _, v in point_cloud_info.items() ] ).to( device )

    mid_camera_directions = torch.stack( [ torch.from_numpy(v.v).to(device) for _, v in point_cloud_info.items() ], dim = -1 )  # n_pts x 3
    thr_angle = torch.Tensor( [ v.theta for _, v in point_cloud_info.items() ] ).to(device)  # n_pts
    thr_cos_angle = torch.cos( thr_angle )  # n_pts

    return thr_d_low, thr_d_up, mid_camera_directions, thr_cos_angle

@my_timeit
def compute_visibility_mask_torch( p3D, ctr_array, thr_low, thr_up, mid_camera_directions, thr_cos_angle, device = "cuda",
                                   curr_rt=None, c_gt=None, R_gt=None ):
    # dist_matrix = np.array( [ np.linalg.norm( p3D - ctr, axis = 1 ) for ctr in ctr_array ] )  # dim: angles x n_pts  # for loop, dist
    # diff_matrix = np.array( [ p3D - ctr for ctr in ctr_array ] )  # dim: angles x n_pts x 3  # for loop, diff

    diff_matrix = p3D[None] - ctr_array  # dim: angles x 3 x n_pts # broadcasting
    dist_matrix = torch.linalg.norm( diff_matrix, dim = 1 )  # dim: angles x n_pts

    dist_matrix_mask = torch.logical_and( dist_matrix > thr_low, dist_matrix < thr_up )  # n_angle x n_pts


    match_directions = diff_matrix / dist_matrix[ :, None, : ]
    cos_angle = torch.sum( match_directions * mid_camera_directions[None], dim = 1 )  # cos_similarity
    angle_mask = cos_angle > thr_cos_angle

    visibility_mask = torch.logical_and( dist_matrix_mask, angle_mask )

    return visibility_mask  # TODO: check angle filtering
    # return dist_matrix_mask



def verify_visibility( C, point_info ):
    difference = C - point_info.xyz
    difference_norm = math.sqrt( sum( difference ** 2 ) )
    cos_angle = difference.dot( point_info.v ) / difference_norm
    cmp_angle = point_info.theta / 2

    if np.arccos( cos_angle ) < cmp_angle:
        return True
    return False


def check_projection(p3D, intrinsics, point_cloud_info, rotmat=None, C=None, img_path=None):

    # for angle_idx in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
    #     plt.plot(p2D[angle_idx][1], p2D[angle_idx][0], 'o')
    #     plt.imshow(img)
    #     plt.show()

    p3D_h = np.hstack( [ p3D, np.ones( p3D.shape[ 0 ] )[ :, None ] ] )

    if rotmat is None:
        rotmat = np.array( [ [ 0.10444, 0.99451, 0.00559 ],
                             [ 0.01538, 0.00401, -0.99987 ],
                             [ -0.99441, 0.10451, -0.01487 ] ] )

        C = np.array( [ -154.07818, -1112.39395, 24.21650 ] )

        img_path = '/home/valentinas98/repos/semantic-matching-for-long-term-visual-localization/data/Extended-CMU-Seasons/slice25/query/img_08436_c1_1292959671932051us.jpg'

    tvec = - rotmat.dot( C )

    p = np.hstack( [ rotmat, tvec[ :, None ] ] )  # 3 x 4

    p2D_h_gt = intrinsics @ p @ p3D_h.T  # 3x3 3x4 4xN
    p2D_gt = p2D_h_gt / p2D_h_gt[ -1:, : ]  # 3xN
    p2D_gt = p2D_gt[ :2, : ]  # 2xN

    visibility_mask = compute_visibility_mask( p3D, C[None], point_cloud_info )

    p2D_cv = cv.projectPoints( p3D, cv.Rodrigues( rotmat )[ 0 ],
                               tvec,
                               intrinsics,
                               np.zeros(4) )[ 0 ].reshape((p3D.shape[0], 2) )

    # region visibility
    intermediate_project_points_ids = [ ]
    intermediate_project_points_idx = []

    X_coords = [ -171.27812, -1123.85325, 21.82090 ]
    R = 4.974444535020842

    import pandas as pd

    pid_df  =   pd.DataFrame( {'pid':list(point_cloud_info.keys())} )

    for pid, info in point_cloud_info.items():
        if (info.xyz[ 0 ] - X_coords[ 0 ]) ** 2 + (info.xyz[ 1 ] - X_coords[ 1 ]) ** 2 < R ** 2:
            intermediate_project_points_ids.append( pid )

    intermediate1_project_points_ids = [ ]

    for pid in intermediate_project_points_ids:
        pt = point_cloud_info[ pid ]
        ssd = np.sum( (C - pt.xyz) ** 2 )
        if pt.dlow ** 2 <= ssd <= pt.dup ** 2:
            intermediate1_project_points_ids.append( pid )

    project_points_ids = [ ]
    for pid in intermediate1_project_points_ids:
        if verify_visibility( C, point_cloud_info[ pid ] ):
            project_points_ids.append( pid )
    # endregion
    project_points_ids = np.asarray( project_points_ids )

    pid_df  =   pid_df[ pid_df.pid.isin( project_points_ids ) ]

    visibility_original = pid_df.index

    img = cv.imread( img_path )

    plt.imshow( img )
    tt_x = p2D_gt[ 0 ][visibility_mask[0]]
    tt_y = p2D_gt[ 1 ][visibility_mask[0]]
    tt_x = np.where( tt_x >= 0, tt_x, 0 )
    tt_y = np.where( tt_y >= 0, tt_y, 0 )
    tt_x = np.where( tt_x <= 1200, tt_x, 0 )
    tt_y = np.where( tt_y <= 800, tt_y, 0 )
    plt.plot( tt_x, tt_y, 'o' )
    plt.show()


def compute_projections_old( X_coords, R, z0, g_direction, x_dir, point_cloud_info, intrinsics,
                             dist_coefs, n_angles = 360 ):
    # X_coords: 3 x n_matches
    # R: 1 x n_matches
    p_array = np.zeros( (n_angles, 3, 4) )
    ctr_array = np.zeros( (n_angles, 3) )

    for phi in range( n_angles ):  # inspect all the angles
        C = np.array( [ X_coords[ 0 ] + R * np.cos( phi * np.pi / 180 ),
                        X_coords[ 1 ] + R * np.sin( phi * np.pi / 180 ),
                        z0 ] )
        ctr_array[phi] = C

        rotmat = get_rotation_matrix( g_direction, x_dir, C, X_coords )
        tvec = - rotmat.dot( C )
        p = np.hstack( [ rotmat, tvec[ :, None ] ] )  # projection matrix
        p_array[ phi ] = p

    p3D = np.vstack( [ v.xyz for _, v in point_cloud_info.items() ] )
    p3D_h = to_homogeneous(p3D)
    p2D_h = intrinsics @ p_array @ p3D_h.T




    # check_projection(p3D, intrinsics, point_cloud_info)
    p2D = p2D_h / p2D_h[ :, -1:, : ]
    # TODO add dist coeff
    p2D = p2D[ :, :2, : ]  # dim: (360, 2, 92916) (angles, xy, n_pts)

    return p3D, p2D, ctr_array

@my_timeit
def _get_rotation_matrices_for_centers( C, phi_range, g_direction, x_direction, match_3d_coords ):
    p_array = np.zeros( (len( phi_range ), 3, 4) )

    for phi in phi_range:  # inspect all the angles
        rotmat = get_rotation_matrix( g_direction, x_direction, C[phi], match_3d_coords )
        tvec = - rotmat.dot( C[phi] )
        p = np.hstack( [ rotmat, tvec[ :, None ] ] )  # projection matrix
        p_array[ phi_range ] = p

    return p_array

@my_timeit
def compute_projections( X_coords, ctr_array, phi, g_direction, x_dir, p3D, intrinsics, dist_coefs, n_angles = 360,device="cuda"):
    '''Version vectorising the computation for all temptative cameras associated to one match
    :param ctr_array:
    '''


    rt_array = _get_rotation_matrices_for_centers( ctr_array, phi, g_direction, x_dir, X_coords )
    p3D_h = to_homogeneous(p3D)  # TODO: fix


    intrinsics = torch.from_numpy( intrinsics ).to( device )
    p2p1 = torch.from_numpy( dist_coefs[ :-3:-1 ].copy() ).to( device )
    dist_coefs = torch.from_numpy( dist_coefs ).to( device )
    rt_array = torch.from_numpy( rt_array ).to( device )
    p3D_h = torch.from_numpy( p3D_h ).to( device )

    p2D_h = rt_array @ p3D_h.T  # (7000, 360, 3, 4) @ (4, 93000)

    # check_projection(p3D, intrinsics, point_cloud_info)

    p2D = p2D_h[ :, :-1 ] / p2D_h[ :, -1: ]

    # region Compute distortions
    # To calculate the distorted image coordinate (u,v) of a point in camera coordinates (X, Y, Z):

    # x_d = x( 1 + k1 * r ^ 2 + k2 * r ^ 4 ) + 2 * p1 * x * y + p2 * (r ^ 2 + 2 * x ^ 2)
    #                           y_d = y(1 + k1*r^2 + k2*r^4) + 2*p2*x*y + p1*(r^2 + 2*y^2)
    #                     where r^2 = x^2 + y^2
    #                             x = X/Z
    #                             y = Y/Z

    #  |u|     |fx 0 cx|     |x_d|
    #  |v|  =  |0 fy cy|  *  |y_d|
    #  |w|     |0  0  1|     | 1 |

    # (Since w is 1 in this case we don't need to normalize u and v).
    #
    r2  =   torch.sum( torch.square( p2D ), dim = 1 )[:, None]
    r4  =   torch.square(r2)

    cdist =   1 + dist_coefs[ 0 ] * r2 + dist_coefs[ 1 ] * r4
    xy_2  =   2 * (p2D[:, 0, :] * p2D[:, 1, :])[:, None]

    #   dist_coefs = [ k1, k2, p1, p2 ]
    p2D_d   =   p2D * cdist\
                + dist_coefs[ 2: ][None, :, None]* xy_2 \
                + p2p1[None, :, None] * ( r2 + 2 * torch.square( p2D ) )
    # endregion

    p2D_dh = torch.concatenate( [ p2D_d, torch.ones_like( r2 ) ], dim = 1 )
    p2D_dh = intrinsics @ p2D_dh  # (3, 3) @ (7000, 360, 3, 93000)

    p2D = p2D_dh[ :, :2, : ]  # dim: (360, 2, 92916) (angles, xy, n_pts)

    return p2D.cpu(), rt_array.cpu()

@my_timeit
def compute_projections_torch( rt_array, p3D, intrinsics, dist_coefs, device="cuda"):
    '''Version vectorising the computation for all temptative cameras associated to one match
    :param ctr_array:
    '''


    p3D_h = to_homogeneous_torch( p3D, coord_axis = 0, device = device )  # (4, 92916)


    p2p1 = torch.Tensor( [ dist_coefs[ -1 ], dist_coefs[ -2] ] ).to( device )


    p2D_h = rt_array @ p3D_h  # (7000, 360, 3, 4) @ (4, 93000)

    # check_projection(p3D, intrinsics, point_cloud_info)

    p2D = p2D_h[ :, :-1 ] / p2D_h[ :, -1: ]

    # region Compute distortions
    # To calculate the distorted image coordinate (u,v) of a point in camera coordinates (X, Y, Z):

    # x_d = x( 1 + k1 * r ^ 2 + k2 * r ^ 4 ) + 2 * p1 * x * y + p2 * (r ^ 2 + 2 * x ^ 2)
    #                           y_d = y(1 + k1*r^2 + k2*r^4) + 2*p2*x*y + p1*(r^2 + 2*y^2)
    #                     where r^2 = x^2 + y^2
    #                             x = X/Z
    #                             y = Y/Z

    #  |u|     |fx 0 cx|     |x_d|
    #  |v|  =  |0 fy cy|  *  |y_d|
    #  |w|     |0  0  1|     | 1 |

    # (Since w is 1 in this case we don't need to normalize u and v).
    #
    r2  =   torch.sum( torch.square( p2D ), dim = 1 )[:, None]
    r4  =   torch.square(r2)

    cdist =   1 + dist_coefs[ 0 ] * r2 + dist_coefs[ 1 ] * r4
    xy_2  =   2 * (p2D[:, 0, :] * p2D[:, 1, :])[:, None]

    #   dist_coefs = [ k1, k2, p1, p2 ]
    p2D_d   =   p2D * cdist\
                + dist_coefs[ 2: ][None, :, None]* xy_2 \
                + p2p1[None, :, None] * ( r2 + 2 * torch.square( p2D ) )
    # endregion

    p2D_dh = torch.concatenate( [ p2D_d, torch.ones_like( r2 ) ], dim = 1 )
    p2D_dh = intrinsics @ p2D_dh  # (3, 3) @ (7000, 360, 3, 93000)

    p2D = p2D_dh[ :, :2, : ]  # dim: (360, 2, 92916) (angles, xy, n_pts)

    return p2D


@my_timeit
def get_match_specific_params( match_2d_hcoords, match_3d_coords, g_direction, z0, camera_internals, n_angles ):

    # x_direction is the normalized (3D) viewing direction of the match, in camera coordinates
    x_direction = np.linalg.inv( camera_internals ).dot( match_2d_hcoords )
    x_direction = x_direction / np.linalg.norm( x_direction )
    alpha = np.arccos( np.array( g_direction ).dot( x_direction ) ) - np.pi / 2 # TODO: adding normalization. Verify it is due
    R = np.abs( match_3d_coords[ 2 ] - z0 ) / np.abs( np.tan( alpha ) )

    phi = np.arange( n_angles )
    ctr_x = match_3d_coords[ 0 ] + R * np.cos( phi * np.pi / 180 )
    ctr_y = match_3d_coords[ 1 ] + R * np.sin( phi * np.pi / 180 )
    ctr_z = np.full( n_angles, z0 )
    ctr_array = np.stack( [ ctr_x, ctr_y, ctr_z ], axis = -1 )

    return x_direction, alpha, R, phi, ctr_array

@my_timeit
def extract_point_cloud_coords( point_cloud_info ):
    ## Function 'extract_point_cloud_coords' executed in 4.262584s
    return np.stack( [ v.xyz for _, v in point_cloud_info.items() ], axis=0 )

@my_timeit
def coarse_filter_point_cloud_coords( R, match_3d_coords, p3d_coords ):
    '''

    :param R: ray of visibility cylinder around 3D point
    :param match_3d_coords: 3D point coords
    :param p3d_coords: all point cloud points coords
    :return: all visible point cloud coords
    '''
    dist_flg = np.sum(np.power((p3d_coords[:, 2] - match_3d_coords[:2, None]), 2), axis=0) < R**2
    return p3d_coords[dist_flg]


def compute_GSMC_score_compiled( match_2d_coords, match_3d_id, z0, g_direction, camera_internals, dist_coefs, query_mask, points_3d_coords, point_cloud_info, n_angles = 360 ):

    match_2d_hcoords = np.ones(3)
    match_2d_hcoords[ :2 ] = match_2d_coords # (3,)

    match_3d_coords = np.array( point_cloud_info[ match_3d_id ].xyz ) #(3,)

    # compute unique parameters that are defined in the paper
    x_direction, alpha, R, phi, ctr_array = get_match_specific_params( match_2d_hcoords, match_3d_coords, g_direction, z0, camera_internals, n_angles )

    # points_3d_coords = coarse_filter_point_cloud_coords( R, match_3d_coords, points_3d_coords )

    print("ciao ci siamo")
    p2D, rt_array = compute_projections( match_3d_coords, ctr_array, phi, g_direction, x_direction, points_3d_coords, camera_internals, dist_coefs, n_angles = n_angles )
    p2D, rt_array = compute_projections( match_3d_coords, ctr_array, phi, g_direction, x_direction, points_3d_coords, camera_internals, dist_coefs, n_angles = n_angles , device='cpu')
    # exit(0)


    visibility_mask = compute_visibility_mask( points_3d_coords, ctr_array, point_cloud_info )

    p2D = np.array( p2D )
    p2D = np.rint( p2D ).astype( np.int32 )  # TODO: vediamo se con int32 o int64 va più veloce

    # Verify which point projections fall into the image
    img_plane_mask = np.logical_and( p2D > np.zeros_like( p2D ),
                                     p2D < np.array( [ width, height ] )[ None, :, None ]
                                     ).all( axis = 1 )

    candidate_match_mask = np.logical_and( visibility_mask, img_plane_mask )

    pc_labels = np.hstack( [ v.semantic_label for _, v in point_cloud_info.items() ] )
    # query_mask

    candidate_match_mask = np.where(candidate_match_mask)
    pid_mask = candidate_match_mask[1]
    center_mask = candidate_match_mask[0]
    pc_labels = pc_labels[ pid_mask ]

    visible_pts_coords = p2D[ center_mask, :, pid_mask ]
    visible_pts_labels = query_mask[ visible_pts_coords[:, 1], visible_pts_coords[:, 0] ]
        #
        # [ query_mask[ p2D[ phi ][ 1 ][ candidate_match_mask[ phi ] ],
        #                                p2D[ phi ][ 0 ][ candidate_match_mask[ phi ] ] ]
        #                    for phi in range( n_angles ) ]

    equal_labels = (pc_labels == visible_pts_labels)

    scores = np.bincount( center_mask, weights = equal_labels )
    best_center = np.unique(center_mask)[np.argmax(scores)]
    return scores.max(), visibility_mask[best_center].sum(), rt_array[best_center], best_center




def GSMC_score( p2Dj, p3D_id, point_cloud_info, g_direction, camera_internals, dist_coefs, query_mask, slicepath,
                z0 = None,
                z0_dict = None, exact_filtering = True, use_covisibility = False, img_data = None ):
    """
    Computation of the Geometric Semantic Match Consistency score for the match p2D-p3D by projecting the visible points onto the image plane.
    We implement two versions: one that is closely based on the work of [Toft '17], and one which does not re-compute visible points at every pose, but rather chooses based on covisibility with the considered match.

    p2D: the coordinates of the query image matched point in the image reference system
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
    bestC = [ ]
    tot_points_count = 0
    bestR = [ ]
    bestPointIds = [ ]

    x_dir = np.linalg.inv( camera_internals ).dot( np.array( [ p2Dj[ 0 ], p2Dj[ 1 ], 1 ] ) )

    alpha = np.arccos( np.array( g_direction ).dot( x_dir ) ) - np.pi / 2
    X_coords = np.array( point_cloud_info[ p3D_id ].xyz )
    if z0 is None:
        z0 = z0_dict[ str( p3D_id ) ]
    R = np.abs( X_coords[ 2 ] - z0 ) / np.abs( np.tan( alpha ) )
    C = np.array( [ X_coords[ 0 ] + R, X_coords[ 1 ], z0 ] )

    if not ((point_cloud_info[ p3D_id ].dlow * 0.9) ** 2 < np.sum( (C - X_coords) ** 2 ) < (
            point_cloud_info[ p3D_id ].dup * 1.1) ** 2):
        return 0, [ ], [ ], 0, [ ]

    if use_covisibility:
        if img_data == None:
            images_path = slicepath + '/sparse/images.bin'
            img_data = read_images_binary( images_path )
        seen_img_ids = point_cloud_info[ p3D_id ].image_ids
        viewing_direction = point_cloud_info[ p3D_id ].v
        viewing_angle = point_cloud_info[
                            p3D_id ].theta / 2  # hemiangle, as the whole thing is from an extrema to the other

        visibility_info_pids = set()

        for iid in seen_img_ids:
            visibility_info_pids = visibility_info_pids.union(
                [ pt for pt in img_data[ iid ].point3D_ids if pt != -1 ] )

        project_points_ids = list( visibility_info_pids )
        p3D = np.array( [ point_cloud_info[ pid ].xyz for pid in project_points_ids ] )
        sem_labels = np.array( [ point_cloud_info[ pid ].semantic_label for pid in project_points_ids ] )

        if len( project_points_ids ) == 0:
            return 0, [ ], [ ], 0, [ ]

        tot_points_count = len( project_points_ids )

        phi = - max( min( 2 * viewing_angle, np.pi ), np.pi / 6 )
        ref_phi = np.arctan( viewing_direction[ 1 ] / viewing_direction[ 0 ] )

        while phi < max( min( 2 * viewing_angle, np.pi ), np.pi / 6 ):
            C = np.array(
                [ X_coords[ 0 ] + R * np.cos( ref_phi + phi ), X_coords[ 1 ] + R * np.sin( ref_phi + phi ), z0 ] )
            rotmat = get_rotation_matrix( g_direction, x_dir, C, X_coords )
            tvec = - rotmat.dot( C )
            rvec = cv.Rodrigues( rotmat )[ 0 ]

            score = project_point_cloud( rvec, tvec, camera_internals, point_cloud_info,
                                         project_points_ids, query_mask,
                                         distortion_coefs = dist_coefs,
                                         p3D = p3D, sem_labels = sem_labels )

            if score > bestScore:
                bestScore = score
                bestC = C
                bestR = rotmat

                bestPointIds = project_points_ids

            phi += 1 / 180 * np.pi


    else:

        intermediate_project_points_ids = [ ]

        for pid, info in point_cloud_info.items():
            if (info.xyz[ 0 ] - X_coords[ 0 ]) ** 2 + (info.xyz[ 1 ] - X_coords[ 1 ]) ** 2 < R ** 2:
                intermediate_project_points_ids.append( pid )

        intermediate1_project_points_ids = [ ]

        # Exact filtering cuts at the most extreme observed distances, otherwise leave some tolerance
        if exact_filtering:
            for pid in intermediate_project_points_ids:
                pt = point_cloud_info[ pid ]
                ssd = np.sum( (C - pt.xyz) ** 2 )
                if pt.dlow ** 2 <= ssd <= pt.dup ** 2:
                    intermediate1_project_points_ids.append( pid )
        else:
            for pid in intermediate_project_points_ids:
                pt = point_cloud_info[ pid ]
                ssd = np.sum( (C - pt.xyz) ** 2 )
                if (pt.dlow * 0.9) ** 2 < ssd < (pt.dup * 1.1) ** 2:
                    intermediate1_project_points_ids.append( pid )
        if len( intermediate1_project_points_ids ) < 30:
            return 0, [ ], [ ], len( intermediate_project_points_ids ), [ ]

        phi = 0

        # region Refactoring
        n_angles = 360

        p3D, p2D, ctr_array = compute_projections_old( X_coords, R, z0, g_direction, x_dir, point_cloud_info,
                                                       camera_internals, dist_coefs, n_angles = n_angles )
        visibility_mask = compute_visibility_mask( p3D, ctr_array, point_cloud_info )

        p2D = np.rint( p2D ).astype( np.int32 )  # TODO: vediamo se con int32 o int64 va più veloce

        # Verify which point projections fall into the image
        img_plane_mask = np.logical_and( p2D > np.zeros_like( p2D ),
                                         p2D < np.array( [ width, height ] )[ None, :, None ]
                                         ).all(axis = 1 )

        candidate_match_mask = np.logical_and(visibility_mask, img_plane_mask)

        match_labels = np.hstack( [ v.semantic_label for _, v in point_cloud_info.items() ] )
        # query_mask

        visible_pts_labels = [ query_mask[ p2D[ phi ][ 1 ][ candidate_match_mask[ phi ] ],
                                           p2D[ phi ][ 0 ][ candidate_match_mask[ phi ] ] ]
                               for phi in range( n_angles ) ]

        # TODO: Action items:
        # 1. Aggiungere 4-th dim, confronto con loop solution
        # 2. Check visibilità angoli
        # 3. Calcolare il semantic score
        # 3. Aggiungere le distorsioni
        # -1. Compilare il .py
        # endregion

        while phi < 360:
            C = np.array(
                [ X_coords[ 0 ] + R * np.cos( phi * np.pi / 180 ), X_coords[ 1 ] + R * np.sin( phi * np.pi / 180 ),
                  z0 ] )
            project_points_ids = [ ]

            for pid in intermediate1_project_points_ids:

                if verify_visibility( C, point_cloud_info[ pid ] ):
                    project_points_ids.append( pid )

            if len( project_points_ids ) > 0:

                rotmat = get_rotation_matrix( g_direction, x_dir, C, X_coords )
                tvec = - rotmat.dot( C )
                rvec = cv.Rodrigues( rotmat )[ 0 ]

                #
                #   from scipy. .. import distance_matrix
                #
                #   distance_matrix(v1, v2)

                score = project_point_cloud( rvec, tvec, camera_internals, point_cloud_info, project_points_ids,
                                             query_mask, distortion_coefs = dist_coefs )

                if score > bestScore:
                    bestScore = score
                    bestC = C
                    bestR = rotmat
                    tot_points_count = len( project_points_ids )
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
