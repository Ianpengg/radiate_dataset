import plistlib
import ipdb
import time 
import radiate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from scipy.signal import find_peaks
import rospy
from pylab import *
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from utils.data_class import Box
from utils import data_utils

class_map = {'car': 1, 'bus': 1, 'truck': 1, 'van': 1, 'pedestrian': 2, 'bicycle': 3, 'motorbike': 3, 'other': 4}
    
def frame_to_time(frame, all_timestamp):
    # make a hash table
    str_format = '{:06d}' 
    all_frame = {str_format.format(all_timestamp['frame'][i]): i for i in range(0, len(all_timestamp['frame']))}
    
    #all_frame[frame]
    print(all_frame)

def plot_grid(img, grid_size):
    #new_img = copy(img)
    new_img = img
    grid_size = 24

    height, width, channels = new_img.shape
    for x in range(0, width -1, grid_size):
        cv2.line(new_img, (x, 0), (x, height), (100, 100, 100), 1, 1)
        cv2.line(new_img, (0,x), (height, x), (100, 100, 100), 1, 1)
    cv2.imshow("grid", new_img)
    

    # list = []
    # temp
    # result after transform
    # append to list
    # update tempto result

def main():
    # path to the sequence
    root_path = 'data/radiate/'
    for sequence_name in os.listdir(root_path):
        if sequence_name == "city_1_1":
            # print(sequence_name)
            sequence_path = os.path.join(root_path, sequence_name)
            # print(sequence_path)
        else:
            continue
        # time (s) to retrieve next frame
        dt = 0.25

        # load sequence
        seq = radiate.Sequence(os.path.join(root_path, sequence_name))
        mode = 'train'

        history_scans_num = 5
        if mode == 'train':
            num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
            nsweeps_back = 6  # Number of frames back to the history (including the current timestamp)
            nsweeps_forward = 4  # Number of frames into the future (does not include the current timestamp)
            skip_frame = 0  # The number of frames skipped for the adjacent sequence
            num_adj_seqs = 2
        else:
            num_keyframe_skipped = 1
            nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
            nsweeps_forward = 20
            skip_frame = 0
            num_adj_seqs = 1

        # Get the radar_timestamp
        radar_timestamp = seq.timestamp_radar
        
        # Get the Lidar_timestamp
        lidar_timestamp = seq.timestamp_lidar
        
        # Get the camera timestamp
        camera_timestamp = seq.timestamp_camera

        # Get the gps timestamp
        gps_timestamp = seq.timestamp_gps

        annotation = seq.annotations

        # for object in annotation
        
        
        num_instance  = 0
        
        save_data_dict_list = list() # for storing consecutive sequences; the data consists of timestamps, points, etc
        save_box_dict_list = list() # for storing box annotations in consecutive sequences
        save_instance_token_list = list()
        adj_seq_cnt = 0
        save_seq_cnt = 0  # only used for save data file name


        # Iterate each sample data
        print("Processing scene {} ...".format(sequence_name))
        for radar_idx in range(len(radar_timestamp['frame'])):
            if radar_idx < history_scans_num:
                continue

             # Get the synchronized radar cartesian   
            radar_cart_list = data_utils.get_multiplesweep_bf_radar_idx(sequence_path,
                                                                        annotation,
                                                                        radar_idx, 
                                                                        nsweeps_back,
                                                                        nsweeps_forward)
            #print(len(radar_cart_list))

            save_data_dict = dict()
            box_data_dict = dict()

            #print("radar_idx", radar_idx)
            
            num_instances = 0
            for ann_token in range(1,len(annotation)+1):
                
                instance_boxes = data_utils.get_instance_boxes_multiple_sweep(annotation, radar_idx, ann_token, nsweeps_back, nsweeps_forward)
                    
                # Check if there are sufficent annotated date across the frames
                if instance_boxes is not None:
                    category_name = annotation[ann_token-1]['class_name']
                    flag = False
                    for c, v in class_map.items():
                        if category_name.startswith(c):
                            box_data_dict['category_' + str(ann_token)] = v
                            flag = True
                            break
                    if not flag:
                        box_data_dict['category_' + str(ann_token)] = 4  # Other category
                    # print("object_id", ann_token)
                    #get_instance_box_info(annotation, curr_frame_id,  object_id)

                    assert len(radar_timestamp['frame']) == len(annotation[ann_token-1]['bboxes']) # make sure there are sufficient frame's of annoations

                    box_data = np.zeros((len(instance_boxes), 2 + 2 + 1), dtype=np.float32)
                    box_data.fill(np.nan)


                    for r, box in enumerate(instance_boxes):
                        if box is not None:
                            row = np.concatenate([box.center, box.wh, box.orientation], axis =0)
                            box_data[r] = row[:]
                    #print(box_data)
                    # Save the box data for current instance
                    box_data_dict['instance_boxes_' + str(ann_token)] = box_data
                    num_instances += 1
                else:
                    continue

                # Generate the 2d gt displacement vector

                # if ann_token == 5:
                #     exit()
                # Load Radar data  (polar to cartesian)  (Need to do motion compensation)
            # print(box_data_dict.keys())
            if len(box_data_dict.keys()) == 0:
                #pass
                continue    

            #exit()
            
            output = seq.get_from_timestamp(np.array(radar_timestamp['time'][radar_idx]))
            #cv2.imshow("Radar+Lidar with annotation", data_utils.get_radar_label_from_t(output, seq))
            

            # radar_cart_list[0][radar_cart_list[0] > 0.28] = 1
            # radar_cart_list[0][radar_cart_list[0] < 0.28] = 0
            
            str_format='{:06d}'
            radar_cart_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_idx) + '.png')
            radar_polar_filename = os.path.join(sequence_path, 'Navtech_Polar', str_format.format(radar_idx) + '.png')
            
            radar_polar =  cv2.imread(radar_polar_filename)
            radar_cart = cv2.imread(radar_cart_filename)
            radar_cart_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_idx+1) + '.png')
            radar_cart_2 = cv2.imread(radar_cart_filename)

            radar_cart[:,:,0][radar_cart[:,:,0] >55] = 255
            radar_cart[:,:,0][radar_cart[:,:,0] <55] = 0
            radar_cart[:,:,1][radar_cart[:,:,1] >55] = 255
            radar_cart[:,:,1][radar_cart[:,:,1] <55] = 0
            radar_cart[:,:,2][radar_cart[:,:,2] >55] = 255
            radar_cart[:,:,2][radar_cart[:,:,2] <55] = 0
            radar_cart = cv2.GaussianBlur(radar_cart, (5, 5), 0.1)
            radar_w_annotation = data_utils.get_radar_label_from_t(output, seq)
            concat =  np.hstack([data_utils.get_radar_label_from_t(output, seq), radar_cart])
            concat = cv2.resize(concat, (1800, 900))
            cv2.imshow("Concat",concat)
            #cv2.imshow("Radar_thres", radar_cart_list[1])
            
            plot_grid(radar_w_annotation, 24)
            
            center = (int(1152/2) -360, int(1152/2) + 360)
            #cv2.imshow('Crop', radar_w_annotation[center[0] : center[1], center[0]:center[1],:]) 
            
            cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            
            
            # Generate the ground truth of aggregate 5 frames 
            # return the motion_segmentation , disp_vector, class_result
            # Save the ground-truth and training data
            # exit()



if __name__ == "__main__": 
    main()
    #gps_visualize()
    #odom_visualize()