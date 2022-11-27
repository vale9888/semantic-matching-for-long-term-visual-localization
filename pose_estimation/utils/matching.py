import cv2 as cv
import numpy as np


def find_k_matches(des_scene, des_query, k=2):
    """Iterate over the 2D image points, and for each of them we return the k nearest neighbors."""

    # Info Flann
    #  algorithm=1 means we use the KDTREE algorithm
    #  trees=5 means 5 trees are used by KDTREE to search
    #  checks=50 is the number of times the trees in the index should be recursively traversed
    flann_matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    nearest_matches = flann_matcher.knnMatch(des_query.astype(np.float32), des_scene.astype(np.float32), k=k)

    return tuple(nearest_matches)


def find_two_exclusive_matches(des_scene, des_query, point3D_ids):
    """Returns first and second neighbor descriptors so that they do not represent the same point."""
    # Info Flann
    #  algorithm=1 means we use the KDTREE algorithm
    #  trees=5 means 5 trees are used by KDTREE to search
    #  checks=50 is the number of times the trees in the index should be recursively traversed
    flann_matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    # two_nearest_matches = flann_matcher.knnMatch(des_scene.astype(np.float32), des_query.astype(np.float32), k=2)
    two_nearest_matches = flann_matcher.knnMatch(des_query.astype(np.float32), des_scene.astype(np.float32), k=2)

    to_retry = []
    for m, n in two_nearest_matches:
        if point3D_ids[m.trainIdx] == point3D_ids[n.trainIdx]:
            to_retry.append(m.queryIdx)  # check

    two_nearest_matches = list(two_nearest_matches)
    to_retry = np.array(to_retry)

    five_nearest_matches = flann_matcher.knnMatch(des_query[to_retry].astype(np.float32), des_scene.astype(np.float32),
                                                  k=5)
    not_matchable = []
    for c, i in enumerate(to_retry):
        pts = five_nearest_matches[c]
        p1 = pts[0]
        p2 = pts[1]
        ix = 1
        while (ix < 5 and point3D_ids[p1.trainIdx] == point3D_ids[p2.trainIdx]):
            p2 = pts[ix]
            ix += 1
        if point3D_ids[p1.trainIdx] != point3D_ids[p2.trainIdx]:
            p1.queryIdx = i
            p2.queryIdx = i
            if p1.distance < two_nearest_matches[i][0].distance:
                two_nearest_matches[i] = (p1, p2)
            else:
                two_nearest_matches[i] = (two_nearest_matches[i][0], p2)
        else:
            not_matchable.append(i)
    print("Could not find NN for %d descriptors." % (len(not_matchable)))
    return tuple(two_nearest_matches), not_matchable


def ratio_test(two_nearest_matches, lowe_thr):
    good_matches = []
    for m, n in two_nearest_matches:
        if m.distance < lowe_thr * n.distance:
            good_matches.append(m)
    print("[STATUS] Found " + str(len(good_matches)) + " matches validated by the Lowe's ratio test")
    return good_matches


def k_ratio_test(kplus1_nearest_matches, lowe_threshold):
    good_matches_all = []
    for matches in kplus1_nearest_matches:
        good_matches = []
        for m in matches[:-1]:
            if m.distance < lowe_threshold * matches[-1].distance:
                good_matches.append(m)
        if len(good_matches) > 0:
            good_matches_all.append(good_matches)
    return good_matches_all