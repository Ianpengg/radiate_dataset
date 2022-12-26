import ipdb
import time 
import radiate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import argparse
from pylab import sort

import rospy
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from scipy.spatial.transform import Rotation as R
from utils.data_class import Box
from utils import data_utils
from utils.plot_utils import plot_grid
from copy import copy, deepcopy

class_map = {'car': 1, 'bus': 1, 'truck': 1, 'van': 1, 'pedestrian': 2, 'bicycle': 3, 'motorbike': 3, 'other': 4}
    
def frame_to_time(frame, all_timestamp):
    # make a hash table
    str_format = '{:06d}' 
    all_frame = {str_format.format(all_timestamp['frame'][i]): i for i in range(0, len(all_timestamp['frame']))}

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path    


    

def main(save_data):
    # path to the sequence
    root_path = '/media/ee904/Data_stored/radiate/data/radiate'
    

    for sequence_name in os.listdir(root_path):
        print(sequence_name)
        if not sequence_name == "city_1_3":
           continue
        else:
            
            sequence_path = os.path.join(root_path, sequence_name)
        # time (s) to retrieve next frame
        data_save_path = 'data/training/' + sequence_name 
        print(data_save_path)
        data_save_path = check_folder(data_save_path)
        dt = 0.25

        # load sequence
        seq = radiate.Sequence(os.path.join(root_path, sequence_name), display_log=False)
        mode = 'train'

        history_scans_num = 2
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

        # Get the radar odometry result
        
        odom_tf_file  = sequence_path + '/' + sequence_name+ '_tf.txt'

        #extract the tf from the RO result
        odom_dict = data_utils.get_radarodom(odom_tf_file)
        #odom_dict = None # will be use in the future
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
            print("radar", radar_idx)
            # ======================EXPERIMENTAL======================================
            """
             # Get the synchronized radar cartesian   
            radar_cart_list = data_utils.get_multiplesweep_bf_radar_idx(sequence_path,
                                                                        annotation,
                                                                        radar_idx, 
                                                                        nsweeps_back,
                                                                        nsweeps_forward)


            #save_data_dict = dict()
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

                    #get_instance_box_info(annotation, curr_frame_id,  object_id)

                    assert len(radar_timestamp['frame']) == len(annotation[ann_token-1]['bboxes']) # make sure there are sufficient frame's of annoations

                    box_data = np.zeros((len(instance_boxes), 2 + 2 + 1), dtype=np.float32)
                    box_data.fill(np.nan)


                    for r, box in enumerate(instance_boxes):
                        if box is not None:
                            row = np.concatenate([box.center, box.wh, box.orientation], axis =0)
                            box_data[r] = row[:]
                    # Save the box data for current instance
                    box_data_dict['instance_boxes_' + str(ann_token)] = box_data
                    num_instances += 1
                else:
                    continue

                # Generate the 2d gt displacement vector

                # Load Radar data  (polar to cartesian)  (Need to do motion compensation)
            # print(box_data_dict.keys())
            if len(box_data_dict.keys()) == 0:
                #pass
                continue    
             # ======================EXPERIMENTAL======================================
            """
            # print(radar_idx)
            # print(len(radar_timestamp['frame']))
            output = seq.get_from_timestamp(np.array(radar_timestamp['time'][radar_idx]))
            #cv2.imshow("Radar+Lidar with annotation", data_utils.get_radar_label_from_t(output, seq))
            
            cart_radar_list = data_utils.load_radar_data(sequence_path, odom_dict, radar_idx, history_scans_num, cart_resolution=0.17361, image_warp=True)
            #print(len(cart_radar_list))
            
            
            # ====================Debug start =======================================================
            # radar_cart_list[0][radar_cart_list[0] > 0.28] = 1
            # radar_cart_list[0][radar_cart_list[0] < 0.28] = 0
            
            str_format='{:06d}'
            radar_cart_filename = os.path.join(sequence_path, 'Navtech_Cartesian', str_format.format(radar_idx+1) + '.png')
            radar_polar_filename = os.path.join(sequence_path, 'Navtech_Polar', str_format.format(radar_idx+1) + '.png')
            #print(radar_cart_filename)
            radar_polar =  cv2.imread(radar_polar_filename)
            radar_cart = cv2.imread(radar_cart_filename, 0)

            radar_w_annotation = data_utils.get_radar_label_from_t(output, seq, fill=False)
            if np.any(radar_w_annotation == None):
                continue
            #plot_grid(radar_w_annotation, 24)

            center = (int(1152/2) -256, int(1152/2) + 256)
            #cv2.imshow('Crop', radar_w_annotation[center[0] : center[1], center[0]:center[1],:]) 
            crop_img = radar_w_annotation[center[0] : center[1], center[0]:center[1],:]
            
            radar_w_annotation = data_utils.get_radar_label_from_t(output, seq, fill=True)
            crop_img = radar_w_annotation[center[0] : center[1], center[0]:center[1],:]
            final_mask = data_utils.convert_to_mask(crop_img)
            def create_pixel_radar_map(radar_cart, final_mask):
                center = (int(1152/2) -256, int(1152/2) + 256)
                radar_cart = radar_cart[center[0] : center[1], center[0]:center[1]]
                
                combine_img = np.zeros((512,512,3))
                combine_img = np.stack([radar_cart, radar_cart, radar_cart], 2)
                

                radar_cart[radar_cart > 40] = 255
                radar_cart[radar_cart < 40] = 0
                final_mask  = final_mask * radar_cart
                radar_cart[radar_cart > 55] = 255
                radar_cart[radar_cart < 55] = 0
                radar_mask = radar_cart == 255
                viz_gt_rmg = np.zeros((512, 512, 3))
                viz_gt_rmg[:, :, 2] = radar_mask *1
                viz_gt_rmg = viz_gt_rmg.astype(np.float32)

                viz_gt_seg = np.zeros((512, 512, 3))
                viz_gt_seg[:, :, 0] = np.logical_and(radar_mask, final_mask) *255
                viz_gt_seg = viz_gt_seg.astype(np.float32)
                # combine_img[radar_mask,:] = [255, 0, 0]
                # print(radar_mask.shape)

                # combine_img[]
               
                # cv2.imshow("thrres", np.uint8((combine_img*2 +viz_gt_seg+ viz_gt_rmg )/ 2.0))
                # radar_mask = cv2.GaussianBlur((radar_mask).astype(np.float32), (3,3), 0.5)
                # cv2.imshow("original", combine_img)
                # cv2.imshow("pixel", np.uint8(radar_mask*255))
                # cv2.imshow("MASK", final_mask)
                return radar_mask, final_mask
            radar_pixel_map, final_mask = create_pixel_radar_map(radar_cart, final_mask)

            #cv2.imshow("MASK", final_mask)
            #cv2.imshow("radar_map", radar_pixel_map/255.)
            # cv2.imshow('raw_radar', cart_radar_list[0])
            #cv2.waitKey(0)

            def viz_combined(img, denoised_img, motion_seg):
                # print((radar_pixel_map))
                # print(final_mask.dtype)
                viz_img = np.zeros((512,512,3))
                viz_img = np.stack((img,img,img), axis=2)
                viz_denoised_img = np.zeros((512,512,3))
                if 1:
                    viz_denoised_img[:,:,0] = (denoised_img * np.logical_not(motion_seg))
                    viz_seg = np.zeros((512,512,3))
                    viz_seg[:,:,2] = motion_seg
                    return (viz_img*2+viz_denoised_img+viz_seg)/2.
                else:
                    viz_denoised_img[:,:,2] = denoised_img

                    return (viz_img*2+viz_denoised_img)/2.
            #cv2.imshow("test", viz_combined(cart_radar_list[0]/255., radar_pixel_map, final_mask))
            #cv2.destroyAllWindows()
            #cv2.waitKey()
            # ====================Debug end =======================================================
            

            if save_data:
                # Save the training_data
                save_data_dict = dict()
                for i in range(len(cart_radar_list)):
                    save_data_dict['raw_radar_' + str(i)] = cart_radar_list[i]
                save_data_dict['gt_car_mask'] = final_mask
                save_data_dict['gt_radar_pixel'] = radar_pixel_map
                save_file_name = os.path.join(data_save_path + '/', str(radar_idx) + '.npy')
                #print(save_file_name)

                np.save(save_file_name, arr=save_data_dict)
                print(f"Successfully save the file {save_file_name} !")
           

def get_args():
    parser = argparse.ArgumentParser(description='Generate the data for training')
    parser.add_argument('--save', action="store_true", help='Whether to save')

    return parser.parse_args()

if __name__ == "__main__": 
    args = get_args()
    main(args.save)
    