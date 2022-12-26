import radiate 
import numpy as np
import pandas 
import matplotlib.pyplot as plt
import cv2
import os
import rospy

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R

# tf_mat = np.zeros((4,4))
# r = R.from_quat([])

def polar_to_cart(raw_example_data, cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True):


    ##########################################################################################
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
   
    ##########################################################################################
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]

    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img


def get_id(t, all_timestamps, time_offset=0.0):
    """get the closest id given the timestamp

    :param t: timestamp in seconds
    :type t: float
    :param all_timestamps: a list with all timestamps
    :type all_timestamps: np.array
    :param time_offset: offset in case there is some unsynchronoised sensor, defaults to 0.0
    :type time_offset: float, optional
    :return: the closest id
    :rtype: int
    """
    all_timestamps = np.array(all_timestamps)
    idx = np.argmin(np.abs(all_timestamps - t + time_offset))
    return idx, all_timestamps[idx]

def get_sync(target_ts, timestamps):
    """
    get the closest id in timestamps given the target_ts
    :param t: timestamp in seconds
    :
    :return: the closest id
    :rtype: int
    """
    idx = np.argmin(np.abs(timestamps - target_ts))
    return idx, timestamps[idx]

def load_timestamp( timestamp_path):
    """load all timestamps from a sensor

    :param timestamp_path: path to text file with all timestamps
    :type timestamp_path: string
    :return: list of all timestamps
    :rtype: dict
    """
    with open(timestamp_path, "r") as file:
        lines = file.readlines()
        timestamps = {'frame': [], 'time': []}
        for line in lines:
            words = line.split()
            timestamps['frame'].append(int(words[1]))
            timestamps['time'].append(__timestamp_format(words[3]))
    return timestamps

def __timestamp_format(raw_timestamp):
    """
    function to fix the timestamp
    """
    raw_decimal_place_len = len(raw_timestamp.split('.')[-1])
    if(raw_decimal_place_len < 9):
        place_diff = 9 - raw_decimal_place_len
        zero_str = ''
        for _ in range(place_diff):
            zero_str = zero_str + '0'
        formatted_timestamp = raw_timestamp.split(
            '.')[0] + '.' + zero_str + raw_timestamp.split('.')[1]
        return float(formatted_timestamp)
    else:
        return float(raw_timestamp)



# path to the sequence
root_path = '../../radiate/data/radiate/'
sequence_name = 'city_1_0'
dt = 0.25
time_offset = None
seq =  radiate.Sequence(os.path.join(root_path, sequence_name))


rospy.init_node('talker', anonymous=True)

pub = rospy.Publisher('pc', PointCloud2, queue_size=100)
lidar_pub = rospy.Publisher('/lidar', PointCloud2, queue_size=100)

r = rospy.Rate(10)

# get all file name of ex: radar, lidar, camera

lidar_file_list = sorted(os.listdir(os.path.join(os.path.join(root_path, sequence_name), 'velo_lidar')))
radar_cart_file_list = sorted(os.listdir(os.path.join(os.path.join(root_path, sequence_name), 'Navtech_Cartesian')))


# Get all timestamp of each data, including the frame idx
data_root = root_path + sequence_name
radar_folder = data_root + '/Navtech_Cartesian/'
timestamps_path = data_root+'/Navtech_Cartesian.txt'
radar_timestamps = load_timestamp(timestamps_path)
#print(radar_timestamps)

lidar_folder = data_root + '/velo_lidar/'
lidar_timestamps_path = data_root+'/velo_lidar.txt'
lidar_timestamps = load_timestamp(lidar_timestamps_path)

radar_idx = 0

auto_playing_mode = True
init = True
# print(radar_timestamps['time'][0])
# print()
# init_time = radar_timestamps['time'][radar_idx]
# id, time= get_id(init_time, radar_timestamps['time'])
# id, time= get_id(init_time, lidar_timestamps['time'])
# print(id , time)


while (1):
    
    init_time = radar_timestamps['time'][radar_idx]

    lidar_id, lidar_timestamp  = get_id((init_time), lidar_timestamps['time'])

    radar_cart = cv2.imread(os.path.join(os.path.join(root_path, sequence_name), 'Navtech_Cartesian',radar_cart_file_list[radar_idx]), 0) 
    cv2.imshow("radar", radar_cart)
    points = seq.read_lidar(os.path.join(os.path.join(root_path, sequence_name), 'velo_lidar',lidar_file_list[lidar_id])).astype(np.float32)
    #print(points.shape)

    points = points[:,:4]
    header = Header()
    header.stamp = rospy.get_rostime() #rospy.Time.from_sec(curr_t)
    # header.frame_id = "radar"
    header.frame_id = "radar"
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            ]
    pc_msg = point_cloud2.create_cloud(header, fields, points[:,:3])
    lidar_pub.publish(pc_msg)
    if init:
        cv2.waitKey(0)
        init = False
    else:
        if auto_playing_mode:
            radar_idx += 1
            key = cv2.waitKey(10)
            if key == 32: # space
                auto_playing_mode = not auto_playing_mode
            if key == 13: # enter
                break
        else:
            key = cv2.waitKey(0)
        
            if key == 100: # d
                radar_idx += 1
            if key == 97: # a
                radar_idx -= 1
            if radar_idx < 0:
                radar_idx=0
            if key == 32: # space
                auto_playing_mode = not auto_playing_mode

