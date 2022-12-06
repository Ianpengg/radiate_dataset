from typing import Tuple, List, Dict
from pyquaternion import Quaternion
import numpy as np 
import matplotlib.pyplot as plt



class Box():
    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: List[float],
                 label: int = np.nan,
                 name: str = None):
        
        self.center = np.array(center)
        self.wh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.name = name   # category name 

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation, other.orientation)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
       
        return center and wlh and orientation and label 
    def __repr__(self):
        repr_str = 'label: {}, xy: [{:.2f}, {:.2f}], wh: [{:.2f}, {:.2f}], ang(degrees): {:.2f}' \
                   ' name: {}'

        return repr_str.format(self.label, self.center[0], self.center[1], self.wh[0],
                               self.wh[1], self.orientation[0], self.name)
    def __str__(self):
        repr_str = 'label: {}, xy: [{:.2f}, {:.2f}], wh: [{:.2f}, {:.2f}], ang(degrees): {:.2f}' \
                   ' name: {}'

        return repr_str.format(self.label, self.center[0], self.center[1], self.wh[0],
                               self.wh[1], self.orientation[0], self.name)
    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x 

if __name__ == "__main__":

    past_frame_skip = 3
