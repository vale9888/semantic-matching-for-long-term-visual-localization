import collections
import json

import numpy as np
import os
import statistics
import torch
import math

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/valentinas98/repos/Thesis_repo')

from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary, qvec2rotmat, read_points3D_binary, read_fullpoints3D_text, write_fullpoints3D_text

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

FullPoint3D = collections.namedtuple(
    "FullPoint3D", ["id", "xyz", "rgb", "error", "semantic_label", "dlow", "dup", "theta", "v", "image_ids", "point2D_idxs"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def get_point_cloud_info(slicepath):
    '''
    Save a dictionary of FullPoint3D objects, that is Point3D information (as from COLMAP output) with additional semantic and distance information
    The save folder is the same as the points3D file from COLMAP.
    '''

    if os.path.exists(os.path.join(slicepath, 'sparse/fullPoints3D.txt')):
        fullpoints3D = read_fullpoints3D_text(os.path.join(slicepath, 'sparse/fullPoints3D.txt'))
    else:
        # Load information
        points3D = read_points3D_binary(os.path.join(slicepath, 'sparse/points3D.bin'))
        images = read_images_binary(os.path.join(slicepath, 'sparse/images.bin'))

        # if os.path.exists(os.path.join(slicepath, 'semantic_masks/numeric/database/database_predictions.json')):
        #     with open(os.path.join(slicepath, 'semantic_masks/numeric/database/database_predictions.json'), 'r') as f:
        #         predictions_ref_dict = json.load(f)
        # else:
        predictions_ref_dict = dict()
        for name in os.listdir(os.path.join(slicepath, 'semantic_masks/numeric/database/')):
            if name.endswith('.txt'):
                db_mask = []
                with open(os.path.join(slicepath, 'semantic_masks/numeric/database/', name), 'r') as fd:
                    lines = fd.readlines()
                    for line in lines:
                        db_mask.append([int(i) for i in line.split(' ')])
                predictions_ref_dict[name[:-4]+'.jpg'] = db_mask
        print(f'Loaded {len(predictions_ref_dict)} queries. Now creating the full point cloud information')


        # Fill enriched dictionary
        fullpoints3D = {}
        for pid, p in points3D.items():

            semantic_label = []
            d = []
            v = []
            for im_id in p.image_ids:
                name = images[im_id].name
                xy = images[im_id].xys[list(images[im_id].point3D_ids).index(pid)]
                semantic_label.append(predictions_ref_dict[name][round(xy[1])][round(xy[0])])

                center = - np.matmul(images[im_id].qvec2rotmat().T, images[im_id].tvec)
                d.append(np.linalg.norm(center-p.xyz))
                v.append((center-p.xyz)/np.linalg.norm(center-p.xyz))

            # semantic_label = statistics.multimode(semantic_label)[0] # ensures we always have a result
            try:
                semantic_label = statistics.mode(semantic_label)
            except:
                vals, counts = np.unique(semantic_label, return_counts=True)
                index = np.argmax(counts)
                semantic_label = vals[index]

            dlow = np.array(d).min()
            dup = np.array(d).max()
            theta = 0
            for i in range(len(v)-1):
                for j in range(i, len(v)):
                    try:
                        if theta < np.arccos(np.array(v[i]).dot(np.array(v[j]))):
                            theta = np.arccos(np.array(v[i]).dot(np.array(v[j])))
                            vlow = v[i]
                            vup = v[j]
                    except:
                        print("Dot prod %f not in -1, 1" %(np.array(v[i]).dot(np.array(v[j]))))
                        continue
            v = (vlow + vup) / 2
            v = v / np.linalg.norm(v)
            fullpoints3D[pid] = FullPoint3D(id=pid, xyz=p.xyz, rgb=p.rgb, error=p.error, semantic_label=semantic_label,
                                            dlow=dlow, dup=dup, theta=theta, v=v, image_ids=p.image_ids, point2D_idxs=p.point2D_idxs)

        # Save file
        write_fullpoints3D_text(fullpoints3D, os.path.join(slicepath, 'sparse/fullPoints3D.txt'))

    return fullpoints3D


# def get_rotation_matrix_from_axes(axes_before, axes_after):
#     """
#     Given two lists of np.array representing in order x, y, z axes,
#     determine the rotation matrix through SVD.
#     """
#
#     B = np.outer(axes_before[:, 0], axes_after[:,0]) + np.outer(axes_before[:, 1], axes_after[:,1])
#     U, S, V = np.linalg.svd(B)
#     M = np.diag([1,1,np.linalg.det(U)*np.linalg.det(V)])
#
#     return np.matmul(U, np.matmul(M, V.T))


def get_rotation_matrix(g_cam, xj_cam, C, Xj, zero_tol = 1e-3):
    '''Given the information in camera matrix, returns the rotation matrix from the world coordinate system to the camera coordinate system'''
    #The used reference is the camera one because it is the only one that's fully known. We rotate that ref. sys to the world one

    g_cam = g_cam / np.linalg.norm(g_cam)
    xj_cam = xj_cam / np.linalg.norm(xj_cam)

    if 1-np.abs(g_cam.dot(np.array([0, 0, 1]))) < zero_tol:
        beta = 0 if g_cam.dot(np.array([0, 0, 1]))>0 else np.pi
        R1 = np.eye(3)
        R2 = np.array([[1, 0, 0], [0, np.cos(beta), np.sin(beta)], [0, -np.sin(beta), np.cos(beta)]])

    else:
        n_cam = np.cross(g_cam, np.array([0, 0, 1]))
        if np.abs(n_cam.dot(np.array([0,1,0]))) < zero_tol:
            alpha = np.pi/2
        else:
            alpha = np.arctan(n_cam.dot(np.array([ 0, 1, 0]))/ n_cam.dot(np.array([1, 0,0])))
        R1 = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

        beta = np.arccos(-g_cam.dot(np.array([0,0,1])))
        R2 = np.array([[1, 0, 0], [0, np.cos(beta), np.sin(beta)], [0, -np.sin(beta), np.cos(beta)]])

    xj_12 = np.matmul(R2, R1).dot(xj_cam)
    if np.abs(Xj[0]) < zero_tol and np.abs(Xj[1]):
        print("Degenerate config")
        return np.matmul(R2, R1).T

    csi = np.pi + np.arccos(xj_12[:2].dot((C-Xj)[:2])/np.linalg.norm(xj_12[:2])/np.linalg.norm((Xj-C)[:2])) # np.arccos(xj_12[:2].dot((Xj-C)[:2])/np.linalg.norm(xj_12[:2])/np.linalg.norm((Xj-C)[:2])) if xj_12[2] * (C-Xj)[2] > 0 else np.arccos(xj_12[:2].dot((Xj-C)[:2])/np.linalg.norm(xj_12[:2])/np.linalg.norm((Xj-C)[:2]))
    R3 = np.array([[np.cos(csi), np.sin(csi), 0], [-np.sin(csi), np.cos(csi), 0], [0, 0, 1]])

    R = np.matmul(R3, np.matmul(R2, R1))

    if np.abs(np.linalg.det(R)+1)<zero_tol:
        R[:,2] = -R[:,2]

    return R.T

def get_rotation_matrix_nd( g_cam, xj_cam, C, Xj, zero_tol = 1e-3 ):
    '''
    :param g_cam: (3, ) or (3,1) array containing the direction of gravity in camera coordinates
    :param xj_cam: (3, n_pts) array containing homogeneous direction vectors (in rotated world coordinates, that is before applyng camera coordinates transformation) for all considered matches
    :param C: (3, n_angles, n_pts ) array containing all temptative centers coordinates
    :param Xj: (3, n_pts) array containing 3D point coordinates (in the world reference system) for all considered matches
    :return: ( n_angles, n_pts, 3, 3 ) matrix with all rotation matrices
    '''

    g_cam = g_cam / np.linalg.norm( g_cam )
    xj_cam = xj_cam / np.linalg.norm( xj_cam, axis = 0 )[ None ]

    if 1 - np.abs( g_cam.dot( np.array( [ 0, 0, 1 ] ) ) ) < zero_tol:
        beta = 0 if g_cam.dot( np.array( [ 0, 0, 1 ] ) ) > 0 else np.pi
        R1 = np.eye( 3 )
        R2 = np.array( [ [ 1, 0, 0 ],
                         [ 0, np.cos( beta ), np.sin( beta ) ],
                         [ 0, -np.sin( beta ), np.cos( beta ) ] ] )
    else:
        n_cam = np.cross( g_cam, np.array( [ 0, 0, 1 ] ) )
        if np.abs( n_cam.dot( np.array( [ 0, 1, 0 ] ) ) ) < zero_tol:
            alpha = np.pi / 2
        else:
            alpha = np.arctan( n_cam.dot( np.array( [ 0, 1, 0 ] ) ) / n_cam.dot( np.array( [ 1, 0, 0 ] ) ) )
        R1 = np.array(
            [ [ np.cos( alpha ), np.sin( alpha ), 0 ], [ -np.sin( alpha ), np.cos( alpha ), 0 ], [ 0, 0, 1 ] ] )

        beta = np.arccos( -g_cam.dot( np.array( [ 0, 0, 1 ] ) ) )
        R2 = np.array(
            [ [ 1, 0, 0 ], [ 0, np.cos( beta ), np.sin( beta ) ], [ 0, -np.sin( beta ), np.cos( beta ) ] ] )

    R2R1 = (R2 @ R1)  # 3 x 3
    xj_12 = R2R1 @ xj_cam  # match_pts_2d rotated  # 3 x3 @ 3 x 1078

    # TODO: vedere config degenerata sotto
    # if np.abs( Xj[ :, 0 ] ) < zero_tol and np.abs( Xj[ :, 1 ] ):
    #     print( "Degenerate config" )
    #     return R2R1.T

    ctr_matches_directions = (C - Xj[ :, None, : ])[ :2 ]
    ctr_matches_directions = ctr_matches_directions / np.linalg.norm( ctr_matches_directions, axis = 0 )[
        None ]  # (2 x n_angles x n_matches)

    csi = np.pi + np.arccos( np.sum( xj_12[ :2, None, : ] * ctr_matches_directions, axis = 0 ) )

    R3 = np.stack( [
        np.stack( [ np.cos( csi ), np.sin( csi ), np.zeros_like( csi ) ], axis = -1 ),
        np.stack( [ -np.sin( csi ), np.cos( csi ), np.zeros_like( csi ) ], axis = -1 ),
        np.stack( [ np.zeros_like( csi ), np.zeros_like( csi ), np.ones_like( csi ) ], axis = -1 )
    ], axis = -1 )  # 360 x 1078 x 3 x 3

    R = R3 @ R2R1

    # region Fix determinant < 0
    R_det = np.linalg.det( R )
    flag = R_det < 0
    flag_reshaped = flag.reshape( flag.shape[ 0 ] * flag.shape[ 1 ] )
    R_reshaped = R.reshape( R.shape[ 0 ] * R.shape[ 1 ], R.shape[ 2 ], R.shape[ 3 ] )
    R_reshaped[ flag_reshaped, :, -1 ] = - R_reshaped[ flag_reshaped, :, -1 ]  # multiply last colum to -1
    R = R_reshaped.reshape( R.shape[ 0 ], R.shape[ 1 ], R.shape[ 2 ], R.shape[ 3 ] )
    # endregion

    return np.swapaxes( R, -1, -2 )


def get_rotation_matrix_nd_torch( g_cam, xj_cam, C, Xj, zero_tol = 1e-6, device = None ):
    '''
    :param g_cam: (3, ) or (3,1) array containing the direction of gravity in camera coordinates
    :param xj_cam: (3, n_pts) array containing homogeneous direction vectors (in rotated world coordinates, that is before applyng camera coordinates transformation) for all considered matches
    :param C: (3, n_angles, n_pts ) array containing all temptative centers coordinates
    :param Xj: (3, n_pts) array containing 3D point coordinates (in the world reference system) for all considered matches
    :return: ( n_pts,n_angles,  3, 3 ) matrix with all rotation matrices
    '''

    if device is None:
        device = "cuda"

    g_cam = g_cam / torch.linalg.norm( g_cam )
    xj_cam = xj_cam / torch.linalg.norm( xj_cam, dim = 0 )[ None ]

    z_axis_cam = torch.tensor( [ 0, 0, 1 ], device=device, dtype=torch.float32)
    if 1 - torch.abs( g_cam.dot( z_axis_cam ) ) < zero_tol:
        beta = 0.0 if g_cam.dot( z_axis_cam ) > 0 else math.pi
        R1 = torch.eye( 3, device = device )
        R2 = torch.tensor( [ [ 1, 0, 0 ],
                         [ 0, torch.cos( beta ), torch.sin( beta ) ],
                         [ 0, -torch.sin( beta ), torch.cos( beta ) ] ], device=device, dtype=torch.float32 )
    else:
        n_cam = torch.cross( g_cam, z_axis_cam )
        x_axis_cam = torch.tensor( [ 1, 0, 0 ], device=device, dtype=torch.float32 )
        y_axis_cam = torch.tensor( [ 0, 1, 0 ], device=device, dtype=torch.float32 )

        alpha = torch.arctan( n_cam.dot( y_axis_cam ) / ( n_cam.dot( x_axis_cam ) + zero_tol ) ) #add a small eps to avoid division by zero
        R1 = torch.tensor([ [ torch.cos( alpha ), torch.sin( alpha ), 0 ],
                            [ -torch.sin( alpha ), torch.cos( alpha ), 0 ],
                            [ 0, 0, 1 ] ], device=device, dtype=torch.float32 )

        beta = torch.arccos( -g_cam.dot( z_axis_cam ) )
        R2 = torch.tensor( [ [ 1, 0, 0 ],
                             [ 0, torch.cos( beta ), torch.sin( beta ) ],
                             [ 0, -torch.sin( beta ), torch.cos( beta ) ] ], device=device, dtype=torch.float32 )

    R2R1 = (R2 @ R1)  # 3 x 3
    xj_12 = R2R1 @ xj_cam  # match_pts_2d rotated  # 3 x3 @ 3 x 1078

    # TODO: vedere config degenerata sotto
    # if np.abs( Xj[ :, 0 ] ) < zero_tol and np.abs( Xj[ :, 1 ] ) < zero_tol: # this should never happen: the camera would never see a point that is right on its center. Not sure what would be the right return val. I will add an assertion below
    #     print( "Degenerate config" )
    #     return R2R1.T

    ctr_matches_directions = (C - Xj[ :, None, : ])[ :2 ]
    assert torch.sum( torch.abs( ctr_matches_directions ), dim = 1 ).min() > 0, "Attempting to compute rotation from the same with center on a 3D point"
    ctr_matches_directions = ctr_matches_directions / torch.linalg.norm( ctr_matches_directions, dim = 0 )[None ]
    # (2 x n_angles x n_matches)
    xj_12 = ( xj_12[ :2 ] / torch.linalg.norm( xj_12[ :2] , dim = 0 ) )[ :, None ]

    csi = math.pi + torch.arccos( torch.sum( xj_12 * ctr_matches_directions, dim = 0 ) )

    R3 = torch.stack( [
        torch.stack( [ torch.cos( csi ), torch.sin( csi ), torch.zeros_like( csi ) ], dim = -1 ),
        torch.stack( [ -torch.sin( csi ), torch.cos( csi ), torch.zeros_like( csi ) ], dim = -1 ),
        torch.stack( [ torch.zeros_like( csi ), torch.zeros_like( csi ), torch.ones_like( csi ) ], dim = -1 )
    ], dim = -2 )  # 360 x 1078 x 3 x 3

    R = R3 @ R2R1

    # region Fix determinant < 0
    R_det = torch.linalg.det( R )
    flag = R_det < 0
    flag_reshaped = flag.reshape( flag.shape[ 0 ] * flag.shape[ 1 ] )
    R_reshaped = R.reshape( R.shape[ 0 ] * R.shape[ 1 ], R.shape[ 2 ], R.shape[ 3 ] )
    R_reshaped[ flag_reshaped, :, -1 ] = - R_reshaped[ flag_reshaped, :, -1 ]  # multiply last colum to -1
    R = R_reshaped.reshape( R.shape[ 0 ], R.shape[ 1 ], R.shape[ 2 ], R.shape[ 3 ] )
    # endregion

    return torch.movedim(R, (0, 1, 2, 3), (1, 0, 3, 2))
