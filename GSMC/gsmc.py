import cv2 as cv
import numpy as np
import math

from fine_grained_segmentation.utils.file_parsing.read_write_model import read_images_binary
from gsmc_utils import get_rotation_matrix


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
