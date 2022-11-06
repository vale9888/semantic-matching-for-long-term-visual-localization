import math
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


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
