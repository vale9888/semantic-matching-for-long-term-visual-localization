import math
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
import numpy as np

from utils.utils import rgb_to_hex
from pose_estimation.utils.data_loading import get_point_cloud_info, get_ground_truth_poses
from GSMC.gsmc_utils import qvec2rotmat

def show_kp(img_path, pts, spatially_consistent=None, semantically_consistent=None, label=None, out_path=None,
            col=None, alpha=1):
    """
    Plots image/semantic mask and adds detected keypoints.
    Differentiation is available (and independent) for spatial consistence (whether the points have been matched
    according to spatial proximity of camera centres) and semantic consistence (whether the labels in the match coincide).
    """

    print("[STATUS] Found " + str(len(pts)) + " keypoints")
    img = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)


    fig, ax = plt.subplots()

    if col is None:
        col = ['fuchsia' for _ in range(len(pts))]
    else:
        clist = list(col)
        if len(clist) == 1:
            col = [col for _ in range(len(pts))]
    for i, p in enumerate(pts):
        p_x = math.floor(p[0])
        p_y = math.floor(p[1])
        if spatially_consistent is not None and i in spatially_consistent:
            if semantically_consistent is not None and i in semantically_consistent:
                ax.plot(p_x, p_y, 'o', color='white', markersize=4, marker=(5, 2), alpha=alpha)
            else:
                ax.plot(p_x, p_y, 'o', color='white', markersize=2, alpha=alpha)
        else:
            if semantically_consistent is not None and i in semantically_consistent:
                ax.plot(p_x, p_y, 'o', color=col[i], markersize=4, marker=(5, 2), alpha=alpha)
            else:
                ax.plot(p_x, p_y, 'o', color=col[i], markersize=2, alpha=alpha)

    # img_with_kp = cv.drawKeypoints(img, kp, None)
    # plt.imshow(img_with_kp)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path + label + '.png', format='png', dpi=250)
        plt.close()


def show_kp_on_mask(segm_path, pts, spatially_consistent=None, semantically_consistent=None, label=None, out_path=None,
                    col=None):
    if col == None:
        col = 'black'
    show_kp(segm_path, pts, spatially_consistent=spatially_consistent, semantically_consistent=semantically_consistent,
            col=col)


def show_projections_on_mask(coords_proj, labels_proj, query_name, mask_query, ax=None, slicepath=None):
    minima = mask_query.min()
    maxima = mask_query.max()
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    # mapper = cm.ScalarMappable(norm, cmap='nipy_spectral')

    if slicepath is None:
        slicepath =  '/home/valentinas98/repos/Thesis_repo/data/Extended-CMU-Seasons/slice22'


    img_path = os.path.join(slicepath,
       'semantic_masks/query',
        query_name[:-4] + '.png')
    img = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter([round(p[0]) for p in coords_proj], [round(p[1]) for p in coords_proj], c=labels_proj,
               cmap='nipy_spectral', norm=norm, s=2)

    plt.imshow(img, alpha=0.4)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    return ax


def plot_poses(poses, gt_poses, ax=None, ratios=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')

    plot_x = [C[0] for _, C in poses]
    plot_y = [C[1] for _, C in poses]
    plot_z = [Rotation.from_matrix(R).as_euler('zxz', degrees=True)[0] for R, _ in poses]
    ax.scatter3D(plot_x, plot_y, plot_z, s=2, c=ratios, cmap=plt.get_cmap('plasma'))
    R_gt, c_gt = gt_poses
    ax.scatter3D(c_gt[0], c_gt[1], Rotation.from_matrix(R_gt).as_euler('zxz', degrees=True)[0], color='fuchsia', s=4)

    plt.show()
    return ax

def plot_visibility_gt( slicepath, img_names ):


    poses, names = get_ground_truth_poses( img_names, slicepath )


    full3Dpoints = get_point_cloud_info(slicepath)
    p3D = np.vstack( [ v.xyz for _, v in full3Dpoints.items() ] )
    p_colors = np.array( [ rgb_to_hex(v.rgb) for _, v in full3Dpoints.items() ] )


    # seen_points = []
    # plot_col = []
    # for pid, pinfo in full3Dpoints.items():
    #     if np.sum((pinfo.xyz - c)**2) < 25**2:
    #         seen_points.append(pinfo.xyz)
    #         plot_col.append(pinfo.rgb)

    # print(len(seen_points))

    c = np.array( poses[ 4: ] )
    rotmat = [ qvec2rotmat( pose[ :4 ] ) for pose in poses ]

    fig = plt.figure(figsize=(20, 15))
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax = _plot_3d_points( p3D, ax, p_colors, s=1 )
    ax = _plot_3d_points( c.reshape((3, 1)), ax, 'red', s=15)
    ax = _plot_camera( c, rotmat, ax )
    # ax.scatter3D([pt[0] for pt in seen_points], [pt[1] for pt in seen_points], [pt[2] for pt in seen_points],
    #              c=np.array([rgb_to_hex(col) for col in plot_col]), s=1)
    # ax.scatter3D([c[0]], [c[1]], [c[2]],
    #              c='red', s=15)

    plt.show()

def _plot_3d_points( pts_coords, ax, pts_colors=None, **kwargs ):
    '''

    :param pts_coords: (3, n_pts)
    :param ax: image ax
    :param pts_colors: (optional) colors for points, in hex
    :return:
    '''

    pts_colors = pts_colors if not pts_colors is None else 'cornflowerblue'
    ax.scatter3D( pts_coords[0], pts_coords[1], pts_coords[2], c=pts_colors, **kwargs )

    return ax


def _plot_3d_axis( center, dir, ax, axis_len = 10, color = None, **kwargs ):

    ax.plot3D( [ center[0], center[0] + axis_len * dir[0] ],
               [ center[1], center[1] + axis_len * dir[1] ],
               [ center[2], center[2] + axis_len * dir[2] ],
               color = color if not color is None else 'orange'
               )
    return ax

def _plot_camera( center, rot, ax ):

    _plot_3d_points( center.reshape((3,1)), ax, pts_colors = 'red' )
    for i in range(3):
        _plot_3d_axis( center, rot[:, i], color = 'black' if i < 2 else 'red' )

    return ax
    # if len(bestR)>0:
    #     #     bestR = np.array(bestR)
    #     #     camera_z_axis_pred = np.array(bestR).dot(np.array([0, 0, 1]))
    #     #
    #     #     # camera_z_axis = np.array(R_gt).dot(np.array([0,0,1]))
    #     #     # ax.plot3D([bestC[0], bestC[0] + 10 * camera_z_axis_pred[0]], [bestC[1], bestC[1] + 10 * camera_z_axis_pred[1]], [bestC[2], bestC[2] + 10 * camera_z_axis_pred[2]], color='red')
    #     #     # ax.plot3D([bestC[0], bestC[0] + 10 * camera_z_axis[0]], [bestC[1], bestC[1] + 10 * camera_z_axis[1]],
    #     #     #           [bestC[2], bestC[2] + 10 * camera_z_axis[2]], color='cyan')
    #     #     ax.plot3D([bestC[0], bestC[0] + 10 * bestR[0, 0]],
    #     #               [bestC[1], bestC[1] + 10 * bestR[1, 0]],
    #     #               [bestC[2], bestC[2] + 10 * bestR[2, 0]], color='cornflowerblue')
    #     #     ax.plot3D([bestC[0], bestC[0] + 10 * bestR[0, 1]],
    #     #               [bestC[1], bestC[1] + 10 * bestR[1, 1]],
    #     #               [bestC[2], bestC[2] + 10 * bestR[2, 1]], color='cornflowerblue')
    #     #     ax.plot3D([bestC[0], bestC[0] + 10 * R_gt[0, 0]],
    #     #               [bestC[1], bestC[1] + 10 * R_gt[1, 0]],
    #     #               [bestC[2], bestC[2] + 10 * R_gt[2, 0]], color='orange')
    #     #     ax.plot3D([bestC[0], bestC[0] + 10 * R_gt[0, 1]],
    #     #               [bestC[1], bestC[1] + 10 * R_gt[1, 1]],
    #     #               [bestC[2], bestC[2] + 10 * R_gt[2, 1]], color='orange')
    #     #
    #     #     ax.plot3D([bestC[0], bestC[0] + 10 * R_gt[0, 2]],
    #     #               [bestC[1], bestC[1] + 10 * R_gt[1, 2]],
    #     #               [bestC[2], bestC[2] + 10 * R_gt[2, 2]], color='red')
    #     #
    #     #
    return