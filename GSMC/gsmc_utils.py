import collections
import json

import numpy as np
import os
import statistics

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

        if os.path.exists(os.path.join(slicepath, 'semantic_masks/numeric/database/database_predictions.json')):
            with open(os.path.join(slicepath, 'semantic_masks/numeric/database/database_predictions.json'), 'r') as f:
                predictions_ref_dict = json.load(f)
        else:
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