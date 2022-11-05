import math
import numpy as np

def compute_pose_errors(R_gt, R_pred, cvec_gt, cvec_pred):
    '''Returns position and rotation error of a pose'''
    position_error = np.linalg.norm(cvec_gt - cvec_pred)
    rotation_error = math.degrees(np.arccos(0.5*(np.trace(np.matmul(np.linalg.inv(R_gt), R_pred))-1)))
    return position_error, rotation_error