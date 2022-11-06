import cv2 as cv
import numpy as np

MSS_SIZE = 4


def p3p_biased_RANSAC(ransac_dict, scores, num_iterations, r, confidence=0.99):
    g = np.random.default_rng(seed=42)
    # g = cv.RNG_UNIFORM

    pts_3d = np.array(ransac_dict['pts_3d'])
    pts_2d = np.array(ransac_dict['pts_2d'])
    camera_matrix = ransac_dict['camera_matrix']
    dist_coeff = np.array(ransac_dict['dist_coeff'])

    probs = np.array(scores) / np.sum(scores)
    population_size = len(pts_3d)

    assert len(pts_3d) == len(pts_2d)
    assert len(pts_3d) == len(scores)
    assert population_size >= 4

    if population_size == MSS_SIZE:
        return cv.solvePnP(pts_3d, pts_2d, camera_matrix, dist_coeff, flags=cv.SOLVEPNP_P3P)

    best_inlier_count = 0
    best_model = [[], []]
    ransac_success = False
    best_mask = []

    for m in range(num_iterations):
        # region (1) Sample a MSS
        mss_idx = g.choice(population_size, MSS_SIZE, replace=False, p=probs)

        # endregion

        # region (2) Fit model k0, ..., k_n
        success_p3p, rvec_p3p, tvec_p3p = cv.solvePnP(pts_3d[mss_idx], pts_2d[mss_idx], camera_matrix, dist_coeff,
                                                      flags=cv.SOLVEPNP_P3P)
        # endregion

        # region (3) Evaluate model
        if success_p3p:

            # compute model errors
            proj_p2D = cv.projectPoints(pts_3d, rvec_p3p, tvec_p3p, camera_matrix, np.array(dist_coeff))[0].reshape(population_size,2)
            err2 = ((proj_p2D - pts_2d) ** 2).sum(axis=1)

            inlier_mask = np.greater(r ** 2, err2)


            # If the current model is the best model, replace the best model with the current model and update number of iterations
            if np.sum(inlier_mask) > max(best_inlier_count, MSS_SIZE-1) :
                best_inlier_count = np.sum(inlier_mask)

                best_mask = inlier_mask
                best_model[0] = rvec_p3p
                best_model[1] = tvec_p3p

                # since on this part we are not sure and it only affects (I think marginally) computational time, we will leave it as it is
                # update number of iterations
                # ep = (population_size-best_inlier_count)/population_size
                # num = 1 - confidence
                # denom = 1 - (1 - ep)**(MSS_SIZE) # is it right? or should I use 3?
                # if denom < sys.float_info.min:
                #     num_iterations = 0
                #
                # num = np.log(num)
                # denom = np.log(denom)
                # if not(denom>=0 or -num >= num_iterations*(-denom)):
                #     num_iterations = round(num / denom)

        # endregion

    # region (5) Local optimization
    if best_inlier_count > 3:
        inliers_2d = pts_2d[best_mask]
        inliers_3d = pts_3d[best_mask]

        try:
            ransac_success, rvec, tvec = cv.solvePnP(inliers_3d, inliers_2d, camera_matrix,
                                                                       dist_coeff, cv.SOLVEPNP_EPNP)
        except:
            print("Exception fitting final model")
            # ransac_success, rvec, tvec = cv.solvePnP(inliers_3d, inliers_2d, camera_matrix,
            #                                          dist_coeff, cv.SOLVEPNP_P3P)

        if ransac_success:
            best_model[0] = rvec
            best_model[1] = tvec
    # endregion

    return ransac_success, best_model[0], best_model[1], best_mask


def p3p_robust_biased_sampling_and_consensus(ransac_dict, scores, num_iterations, r, score_threshold=0.1):
    g = np.random.Generator(np.random.PCG64())

    pts_3d = np.array(ransac_dict['pts_3d'])
    pts_2d = np.array(ransac_dict['pts_2d'])
    camera_matrix = ransac_dict['camera_matrix']
    dist_coeff = np.array(ransac_dict['dist_coeff'])

    probs = np.array(scores) / np.sum(scores)
    population_size = len(pts_3d)

    assert len(pts_3d) == len(pts_2d)
    assert len(pts_3d) == len(scores)
    assert population_size >= 4

    if population_size == MSS_SIZE:
        return cv.solvePnP(pts_3d, pts_2d, camera_matrix, dist_coeff, flags=cv.SOLVEPNP_P3P)

    best_inlier_count = 0
    best_model = [[], []]
    ransac_success = False
    best_mask = []

    for m in range(num_iterations):
        # region (1) Sample a MSS
        mss_idx = g.choice(population_size, MSS_SIZE, replace=False, p=probs)
        # endregion

        # region (2) Fit model k0, ..., k_n
        success_p3p, rvec_p3p, tvec_p3p = cv.solvePnP(pts_3d[mss_idx], pts_2d[mss_idx], camera_matrix, dist_coeff,
                                                      flags=cv.SOLVEPNP_P3P)
        # endregion

        # region (3) Evaluate model
        if success_p3p:

            proj_p2D = cv.projectPoints(pts_3d, rvec_p3p, tvec_p3p, camera_matrix, np.array(dist_coeff))[0].reshape(
                population_size, 2)
            err2 = ((proj_p2D - pts_2d) ** 2).sum(axis=1)

            inlier_mask = np.greater(r ** 2, err2)

            # If the current model is the best model, replace the best model with the current model
            if np.sum(inlier_mask * scores) > best_inlier_count and np.sum(inlier_mask) > MSS_SIZE-1:
                best_inlier_count = np.sum(inlier_mask * scores)
                best_mask = inlier_mask
                best_model[0] = rvec_p3p
                best_model[1] = tvec_p3p

                #   (5) Local optimization
                # success_lo, rvec_lo, tvec_lo = cv.solvePnP(pts_3d[inlier_mask], pts_2d[inlier_mask], camera_matrix,
                #                                            dist_coeff)
                # best_mask = inlier_mask
                # if success_lo:
                #     best_model = [rvec_lo, tvec_lo]
                #     ransac_success = True
                # else:
                #     best_model = [rvec_p3p, tvec_p3p]
                #     ransac_success = True
        # endregion


    # region (4) Fit final model
    best_mask = np.logical_and(best_mask, np.array(scores)>score_threshold)

    if np.sum(best_mask)>3:
        inliers_2d = pts_2d[best_mask]
        inliers_3d = pts_3d[best_mask]

        try:
            ransac_success, best_model[0], best_model[1] = cv.solvePnP(inliers_3d, inliers_2d, camera_matrix, dist_coeff, cv.SOLVEPNP_EPNP)
        except:
            print("Exception fitting final model")

    # endregion

    return ransac_success, best_model[0], best_model[1], best_mask
