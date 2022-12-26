import math
import numpy as np 


 
def GetRotMatFromTransMat(trans_mat) -> np.ndarray:
    """
    Get roll pitch yaw from the 4x4 transformation matrix
    :param trans_mat: 
    :return np.float, np.float, np.float
    """

    R = trans_mat[:3, :3]


    roll  = math.atan2(R[2][1], R[2][2])
    pitch = math.atan2(-R[2][0], np.sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2]))
    yaw   = -math.atan2(R[1][0], R[0][0])

    x = trans_mat[0, 3] 
    y = trans_mat[1, 3]
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])

    return tf_mat, yaw




def GetMatFromXYYaw(x, y, yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat


def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box. (3D)
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask