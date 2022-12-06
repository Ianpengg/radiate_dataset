import radiate
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import rospy

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R

class dataset:
    def __init__(self, seq_name, dt=0.25) -> None:
        self.seq_name = None
        self.seq = None
        self.dt = None
        self.load_seq(seq_name, dt)
        
    def load_seq(self, seq_name, dt=0.25):
        root = os.path.abspath(".")
        self.seq_name = seq_name
        seq_file = os.path.join('data/radiate', seq_name)
        abs_path = os.path.join(root, seq_file)
        self.seq = radiate.Sequence(abs_path)
        self.dt = dt
        #self.output = self.seq.get_from_timestamp()

    def get_lr_from_t(self, t):
        output = self.seq.get_from_timestamp(t)
        sensor_data = output['sensors']
        radar_c = sensor_data['radar_cartesian']
        lidar_c = sensor_data['lidar_bev_image']
        annos = output['annotations']
        #lidar_annos = annos['lidar_bev_image']
        radar_annos = annos['radar_cartesian']
        #print(radar_annos)
        img = np.zeros(radar_c.shape, dtype='int8')
        radar_annos_vis = self.seq.vis(radar_c, radar_annos)
        img = radar_annos_vis

        # lidar_mask = lidar_c[:, :, 0] > 0
        # img[lidar_mask, 2] = 255

        return img

    def save(self):
        if not os.path.exists('save_imgs'):
            os.mkdir('save_imgs')

        # play sequence
        i = 0
        for t in np.arange(self.seq.init_timestamp, self.seq.end_timestamp, self.dt):
            img = self.get_lr_from_t(t)
            cv2.imwrite('save_imgs/' + f'img_{i}.png', img)
            i += 1
    def get_stereo_img(self, t):
        output = self.seq.get_from_timestamp(t)
        sensor_data = output['sensors']
        annos = output['annotations']
        right_bb = self.seq.vis_3d_bbox_cam(sensor_data['camera_right_rect'], annos['camera_right_rect'])
        return right_bb

    def plot(self):
        seq = self.seq
        dt = self.dt

        timestamp_list = np.arange(seq.init_timestamp, seq.end_timestamp, dt)
        radar_timestamp = self.seq.get_sensor_timestamp("radar")
        #print(radar_timestamp)
        idx = 0

        auto_playing_mode = True
        init = True

        while (1):
            radar_cart = self.get_lr_from_t(np.array(radar_timestamp['time'][idx]))
            cv2.imshow("radar", radar_cart)
            right_camera = self.get_stereo_img(np.array(radar_timestamp['time'][idx]))
            
            cv2.imshow('camera right', right_camera)
            # points = seq.read_lidar(os.path.join(os.path.join(root_path, sequence_name), 'velo_lidar',lidar_file_list[idx])).astype(np.float32)
            #print(points.shape)

            # points = points[:,:4]
            # header = Header()
            # header.stamp = rospy.get_rostime() #rospy.Time.from_sec(curr_t)
            # # header.frame_id = "radar"
            # header.frame_id = "radar"
            # fields = [PointField('x', 0, PointField.FLOAT32, 1),
            #         PointField('y', 4, PointField.FLOAT32, 1),
            #         PointField('z', 8, PointField.FLOAT32, 1),
            #         ]
            # pc_msg = point_cloud2.create_cloud(header, fields, points[:,:3])
            # lidar_pub.publish(pc_msg)
            if init:
                cv2.waitKey(0)
                init = False
            else:
                if auto_playing_mode:
                    idx += 1
                    key = cv2.waitKey(10)
                    if key == 13: # enter
                        break
                    if key == 32: # space
                        auto_playing_mode = not auto_playing_mode
                else:
                    key = cv2.waitKey(0)
                
                    if key == 100: # d
                        idx += 1
                    if key == 97: # a
                        idx -= 1
                    if key == 13: # enter
                        break
                    if idx < 0:
                        idx = 0
                    if key == 32: # space
                        auto_playing_mode = not auto_playing_mode
        cv2.destroyAllWindows()
    def test(self, ):
        # output = self.seq.get_from_timestamp(self.seq.end_timestamp)
        # print(output.keys())
        radar_timestamp = self.seq.get_sensor_timestamp("radar")
        #print(radar_timestamp)

def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    else:
        return leaf_layout
rospy.init_node('talker', anonymous=True)

pub = rospy.Publisher('pc', PointCloud2, queue_size=100)
lidar_pub = rospy.Publisher('/lidar', PointCloud2, queue_size=100)

r = rospy.Rate(10)
def main():
    seq_name = 'city_1_1'
    seq = dataset(seq_name, dt=0.25)
    # seq.save()
    #seq.test()
    seq.plot()

        
if __name__ == "__main__":
    main()

