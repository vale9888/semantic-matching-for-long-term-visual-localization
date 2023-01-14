import cv2 as cv
import h5py
import numpy as np
import os
from scipy.spatial import distance_matrix
import math

import fine_grained_segmentation.datasets.dataset_configs as data_configs
from fine_grained_segmentation.utils.file_parsing.read_write_model import read_next_bytes, qvec2rotmat, read_images_binary, read_cameras_binary
from GSMC.gsmc_utils import get_point_cloud_info
from pose_estimation.utils.data_loading import get_ground_truth_poses, get_descriptors_image, get_camera_parameters, _get_reference_images_info_binary
from visualization.visualization_tools import show_kp


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def project_points(Rmat, tvec, camera_internals, point_cloud_info, project_point_ids=None, height=768, width=1024, distortion_coefs=None ):
    if project_point_ids is None:
        project_point_ids = [n for n in point_cloud_info.keys()]

    colors = []
    p3D = np.array([point_cloud_info[pid].xyz for pid in project_point_ids])
    p2D = cv.projectPoints(p3D, Rmat, tvec, camera_internals, np.array(distortion_coefs))[0].reshape(
        (len(project_point_ids), 2))

    for pid in project_point_ids:
        colors.append(point_cloud_info[pid].rgb)

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
                                                            distortion_coefs=dist_coefs)


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
        proj_qkp, proj_qcolors, proj_qpids = project_points(R_gt_query, t_gt_query, intrinsics, full_points3D,
                                                            project_point_ids=visibility_info_pids,
                                                            distortion_coefs=dist_coefs)

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
