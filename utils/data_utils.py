import numpy as np
import os 
import cv2
from .data_class import Box
from .geometric_utils import GetRotMatFromTransMat
import math

def convert_to_mask(img) -> np.ndarray:
    """
    Marked region would be filled with [255,255,255]\n   
    Here we extract the marked region with background filled with [0,0,0] 
    @param img: Image
    @return finak_mask: binary mask
    """
    final_mask = np.zeros((img.shape[:2]))   # shape = (256,256)
    mask = np.where(img == [255, 255, 255]) # mask = (np.array([x,x,x]...), np.array([y,y,y]...), np.array([r,g,b]...) ->[x,y]
    # extract value from the mask to generate the (x ,y) coordinate of the marked region 
    for i in range(0, len(mask[0]),3):
        x = mask[0][i]
        y = mask[1][i]
        final_mask[x][y] = 1
        
    return final_mask

def optical_flow_test(img_fram1, img_fram2):
    img_fram1 = cv2.cvtColor(img_fram1,cv2.COLOR_BGR2GRAY)
    img_fram2 = cv2.cvtColor(img_fram2,cv2.COLOR_BGR2GRAY)
    img_fram1 = img_fram1[int(1152/2) -256: int(1152/2) + 256, int(1152/2) -256: int(1152/2)+256]
    img_fram2 = img_fram2[int(1152/2) -256: int(1152/2) + 256, int(1152/2) -256: int(1152/2)+256]
    hsv = np.zeros((img_fram1.shape[0],img_fram1.shape[1],3)).astype(np.uint8)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(img_fram1,img_fram2, None, 0.5, 5, 3, 5, 5, 1.2, 0) # 6
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    img = rgb.astype(np.float32)/255.
    cv2.imshow("flow", img)


def polar_to_cart(raw_example_data, cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True):

    ##########################################################################################
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    azimuths = np.arange(400) * (math.pi/180)*0.9
    
    
    #azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    raw_example_data = raw_example_data.T
    fft_data = raw_example_data[:, :].astype(np.float32)[:, :, np.newaxis] / 255.
    
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


def check_is_valid(list):
    """
    Check if the sequence of annotations is valid?\n
    Definition of valid:\n
    if whole items in the sequence of annotations are not None
    """
    valid = 0
    for i in list:
        if i is not None:
            valid +=1
    if valid == len(list):
        return True
    else:
        return False

from math import sqrt, cos, radians
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # radius of the earth in km
    x = (radians(lon2) - radians(lon1)) * cos(0.5 * (radians(lat2) + radians(lat1)))
    y = radians(lat2) - radians(lat1)
    d = R * sqrt(x*x + y*y)
    d = d * 1000  # convert to meter
    return d


def get_instance_box_info(annotations, frame_id, object_id):

    instance_id = annotations[object_id - 1]["id"]
    cat_name = annotations[object_id - 1]["class_name"]
    box_info = annotations[object_id - 1]["bboxes"][frame_id - 1]
    if type(box_info) != list:
        #print(type(box_info))
        center = box_info['position'][0:2]
        #print(type(center))
        size = box_info['position'][2:4]
        rotation = [box_info['rotation']]
        # print(center)
        #print(rotation)
        box = Box(center, size, rotation, instance_id, cat_name)
    else:
        return None
    return box

def get_instance_boxes_multiple_sweep(annotations, refer_frame, object_id, nsweeps_back, nsweeps_forward):
    """
    Return the bounding boxes associated with the given object_id.
    The bounding box are across the different sweeps.e
    For this function, the reference sweep is supposed to be from current_frame
    """

    box_list = list()
    
    current_frame = refer_frame
    max_frame_id = len(annotations[object_id -1]['bboxes'])

    for _ in range(nsweeps_back):
        box= get_instance_box_info(annotations, current_frame, object_id)
        
        box_list.append(box)

        if current_frame - 1 < 0:
            break
        else:
            current_frame = current_frame - 1
    
    current_frame = refer_frame
    
    if current_frame + 1 <= max_frame_id:
        
        current_frame = current_frame + 1


        for _ in range(nsweeps_forward):
            box= get_instance_box_info(annotations, current_frame, object_id)
            box_list.append(box)  # It is possible the returned box is None
            
        
            if current_frame + 1 > max_frame_id:
                break
            else:
                current_frame = current_frame + 1
                #print(current_sd_rec)
    
    if check_is_valid(box_list):
        return box_list
    else:
        return None 

def get_radarodom(ro_filename) -> dict:
    """
    Get the radar odometry data from the file   
    Save it with dictionary   
    ex:
    { 'radar_id': tf_mat(4x4), 'radar_id': tf_mat(4x4) ...}
    """
    with open(ro_filename, "r") as file:
        odom_tf_data = {}
        lines = file.readlines()
        for idx in range(len(lines)):
            line = lines[idx]
            tf_data = line.split()
            tf_mat = np.eye(4,4)
            tf_mat[0,:] = np.array(tf_data[:4])
            tf_mat[1,:] = np.array(tf_data[4:8])
            tf_mat[2,:] = np.array(tf_data[8:12])
            tf_mat[3,:] = np.array(tf_data[12:16])

            odom_tf_data[idx] = tf_mat
    return odom_tf_data

def get_radar_label_from_t(output, seq, fill):
    
    if len(output.keys()) > 0:

        sensor_data = output['sensors']
        radar_c = sensor_data['radar_cartesian']
        #lidar_c = sensor_data['lidar_bev_image']
        annos = output['annotations']
        #lidar_annos = annos['lidar_bev_image']
        radar_annos = annos['radar_cartesian']

        img = np.zeros(radar_c.shape, dtype='int8')
        radar_annos_vis = seq.vis(radar_c, radar_annos, fill)
        img = radar_annos_vis
        return img
    else:
        return None
    


def load_radar_data(sequence_path, ro_data, radar_idx, history_scan_num, cart_resolution, image_warp=False):
    from copy import deepcopy
    radar_idx += 1  # fix the start index of array is 0  but file start at 1
    str_format = '{:06d}'
    radar_cart_no_warp_list = list()
    radar_cart_warp_list = list()
    center = (int(1152/2) -256, int(1152/2) + 256)
    
    radar_warp_current = np.zeros((1152,1152))
    radar_warp_prev = np.zeros((1152,1152))
    radar_warp_combine = np.zeros((1152,1152,3))
    for i in range(history_scan_num):
        idx = radar_idx - i     
        radar_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(idx) + '.png')
        radar_cart = cv2.imread(radar_filename, 0)
        if i == 0:
            radar_warp_current = cv2.imread(radar_filename, 0)
        # Motion Compensation part
        if image_warp:
            
            tf_mat = np.identity(3).astype(np.float32)
            #cv2.imshow("original", radar_cart[center[0] : center[1], center[0] : center[1]])
            for j in range(i):
                j = i - 1 - j
                ro_idx = radar_idx - 1 - j
                ro_tf, yaw = GetRotMatFromTransMat(ro_data[ro_idx])
                ro_tf = np.linalg.inv(ro_tf) # 
                tf_mat = tf_mat.dot(ro_tf)
                # print(f'ro_transform:{ro_tf}')
            #trans_radar_cart = radar_cart
            #if i > 0:    
                #trans_radar_cart = rotate_image(radar_cart, yaw)
            
            trans_radar_cart = warp_radar_by_radar_motion(radar_cart, tf_mat, cart_resolution)
            trans_radar_cart = trans_radar_cart[center[0] : center[1], center[0] : center[1]]
            radar_cart_warp_list.append(trans_radar_cart)

            # radar_warp_combine[:, :, 0] = trans_radar_cart / 255.
            # radar_warp_combine[:, :, 2] = radar_warp_current / 255.

            # trans_radar_cart_copy = deepcopy(trans_radar_cart)
            # radar_warp_current_copy = deepcopy(radar_warp_current)

            # #print(trans_radar_cart_copy[500:510, 500:510])
            # #print(radar_warp_current_copy[500:510, 500:510])
            # th = 50
            # trans_radar_cart_copy[trans_radar_cart < th] = 0
            # radar_warp_current_copy[radar_warp_current < th] = 0
            # trans_radar_cart_copy[trans_radar_cart >= th] = 1
            # radar_warp_current_copy[radar_warp_current >= th] = 1

            #radar_warp_combine[:, :, 1] = np.logical_and(trans_radar_cart_copy , radar_warp_current_copy , dtype=np.float32)

            #print(radar_cart)
            #cv2.imshow("warped", radar_warp_combine)
        else:
            radar_cart_no_warp = radar_cart
            radar_cart_no_warp = radar_cart_no_warp[center[0] : center[1], center[0] : center[1]]
            radar_cart_no_warp_list.append(radar_cart_no_warp)
    
    if image_warp:
        #cv2.imshow("warp combined", np.float32(radar_warp_combined) / 5.)
        #cv2.waitKey()
        return radar_cart_warp_list
    else:
        return radar_cart_no_warp_list

def get_multiplesweep_bf_radar_idx(sequence_path: str,
                                   annotations: dict,
                                   radar_idx: int,
                                   nsweeps_back: int = 6,
                                   nsweeps_forward: int = 4):
    """
    Return a radar cartesian image list, The sweeps trace back as well as into the future.
    """
    str_format = '{:06d}'
    radar_cart_list = list()

    max_frame_id = len(annotations[0]['bboxes'])
    current_frame_id = radar_idx 
    
    for i in range(nsweeps_back):
        
        radar_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(i) + '.png')
        radar_cart = cv2.imread(radar_filename)
   
        radar_cart_list.append(radar_cart)
        if current_frame_id - 1 < 0:
            break
        else:
            current_frame_id = current_frame_id - 1
    
    current_frame_id = radar_idx

    if current_frame_id + 1 <= max_frame_id:
        
        current_frame_id = current_frame_id + 1

    for i in range(1, nsweeps_forward + 1):
        radar_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(i) + '.png')
        radar_cart = cv2.imread(radar_filename)
   
        radar_cart_list.append(radar_cart)

        if current_frame_id + 1 > max_frame_id:
            break
        else:
            current_frame_id = current_frame_id + 1


    return radar_cart_list

def warp_radar_by_radar_motion(src, tf_mat, cart_resolution):
    delta_x = tf_mat[0,2]
    delta_y = tf_mat[1,2]
    delta_yaw = np.arctan(tf_mat[1,0] / tf_mat[0,0]) #### !! might have ambiguous issue between pi/2 and -pi/2


    #print("delta_w",  delta_yaw)
    cv_x = -delta_y / cart_resolution
    cv_y = delta_x / cart_resolution
    cv_theta = delta_yaw

    warp_mat = np.zeros((2, 3), dtype=np.float32)
    warp_mat[0, 0] = np.cos(cv_theta)
    warp_mat[0, 1] = np.sin(cv_theta)
    warp_mat[0, 2] = cv_x * np.cos(cv_theta) + cv_y * np.sin(cv_theta) + ( (1 - np.cos(cv_theta)) * src.shape[1] / 2 - np.sin(cv_theta) * src.shape[0] / 2 )
    warp_mat[1, 0] = -np.sin(cv_theta)
    warp_mat[1, 1] = np.cos(cv_theta)
    warp_mat[1, 2] = cv_x * (-np.sin(cv_theta)) + cv_y * np.cos(cv_theta) + (np.sin(cv_theta) * src.shape[1] / 2 + (1 - np.cos(cv_theta)) * src.shape[0] / 2 )
    warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

    return warp_dst

def crop_radar(radar_img, crop_range):
    from copy import deepcopy 
    n, m = radar_img.shape[:2]
    crop_radar = deepcopy(radar_img)
    center = (int(n/2) -crop_range[0], int(m/2) + crop_range[1])
    crop_radar = radar_img[int(m/2) - int(crop_range[1]/2) : int(m/2) + int(crop_range[1]/2), int(n/2) - int(crop_range[0]/2) : int(n/2) + int(crop_range[0]/2)]
        
    return crop_radar
