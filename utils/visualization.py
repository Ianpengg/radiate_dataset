import os
import sys
sys.path.insert(0, '/media/ee904/Data_stored/temp_i/radiate_dataset/')
import radiate 
import pandas as pd
import numpy as np
import data_utils
import matplotlib.pyplot as plt
from pylab import sort



def gps_visualize():
    """
    Use to generate the ego car's gps information (lon, lat)
    And visualize the result gps route

    Optional
    ----------------------------
    save the position_<seq_name>_dict.npy
    save the displacement of ego car between two consecutive frame  
    """
    root_path = 'data/radiate/'
    sequence_name = 'city_1_1'
    sequence_path = root_path + sequence_name
    gps_file = root_path + sequence_name + '/GPS_IMU_Twist'

    save_disp = False
    save_gps = False

    if not os.path.exists(root_path + sequence_name + "/position_" + sequence_name + "_dict.npy"):
        # save gps info as {frame_id: [lon, lat]}
        position_dict = {}
        
        for i in sort(os.listdir(gps_file)):
            frame_id = i.split('.')[0]
            gps_data = pd.read_csv(os.path.join(sequence_path, 'GPS_IMU_Twist',i),delimiter=',', header=None, nrows=1).values
            northing = gps_data[0][0]
            easting = gps_data[0][1]
            position = np.array([northing, easting])
            position_dict[frame_id] = position

        np.save(root_path + sequence_name + "/position_" + sequence_name + "_dict.npy", arr=position_dict)
    
    # Load the processed data_dict of position info
    position_dict_raw = np.load(root_path + sequence_name + "/position_" + sequence_name + "_dict.npy", allow_pickle=True)

    position_dict = position_dict_raw.item()
    key_list = sorted(position_dict.keys())
        
    distance_list = []
    for i, key in enumerate(key_list):

        if i+1 < len(key_list):
            # Check if there are some frame loss, and we need to skip the velocity calculation between this frame and the next frame   => ex: 000001 -> 000050  
            if abs(int(key_list[i]) - int(key_list[i+1])) > 2:
                print("The frame loss begins with the frame:",key_list[i+1])
                continue
    
            if (position_dict[key_list[i]] == position_dict[key_list[i+1]]).all():
                continue
        
            # Calculate the distance between two points in (m)
            lon1, lat1 = position_dict[key_list[i]]
            lon2, lat2 = position_dict[key_list[i+1]]
            
            distance = data_utils.haversine(lon1, lat1, lon2, lat2)
            print("=============Distance=============")
            print(distance)
            distance_list.append(distance)
            
    position_arr = np.zeros((len(position_dict), 2))
    for i, key in enumerate(key_list):
        position_arr[i,:] = (position_dict[key])
    # Save the file 
    if save_disp:
        np.savetxt(root_path + sequence_name + "/" + "disp.txt", np.array(distance_list), fmt='%.4f')
    if save_gps:
        np.savetxt(root_path + sequence_name + "/" + "gps.csv", position_arr)
    f, a = plt.subplots(1, 1, figsize=(21, 9))
    a.scatter(position_arr[:,0], position_arr[:,1], s=1)
    plt.show()

def odom_visualize():
    root_path = 'data/radiate/'
    sequence_name = 'city_1_1'
    sequence_path = root_path + sequence_name
    odom_tf_file  = sequence_path + '/' + 'city_1_1_tf.txt'

    #extract the tf from the RO result
    with open(odom_tf_file, "r") as file:
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

    # Load the processed data_dict of position info
    position_dict_raw = np.load(root_path + sequence_name + "/position_" + sequence_name + "_dict.npy", allow_pickle=True)
    position_dict = position_dict_raw.item()

    key_list = sorted(position_dict.keys())
    init_pose = np.eye(4)
    init_pose[:2,3] = position_dict[key_list[0]]
    print("Init_pose\n",init_pose)
    
    # Store the x, y of the odometry to array
    odom_cart = [init_pose[:2, 3]]
    idx = 0 
    while(idx < len(odom_tf_data)):
        result = np.matmul(init_pose, odom_tf_data[idx])
        odom_cart.append(result[:2, 3])
        init_pose = result
        idx +=1
    
    odom_cart = np.array(odom_cart)
    plt.title("Radar Odometry Result")
    plt.scatter(odom_cart[:,0], odom_cart[:,1], c="b", s=1)
    plt.show()


if __name__ == "__main__":
    #gps_visualize()
    odom_visualize()