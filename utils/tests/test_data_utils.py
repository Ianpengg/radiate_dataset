import unittest
import os 
import cv2
import numpy as np
import sys
sys.path.insert(0, '/media/ee904/Data_stored/temp_i/radiate_dataset/')
import utils.data_utils as data_utils



class TestDataUtils(unittest.TestCase):

    def test_load_radar(self):
        history_scans_num = 2
        cart_resolution = 0.17361
        root_path = 'data/radiate/'
        sequence_name = "city_1_1"
        sequence_path = os.path.join(root_path, sequence_name)

        odom_tf_file  = sequence_path + '/' + sequence_name+ '_tf.txt'
        odom_dict = data_utils.get_radarodom(odom_tf_file)
        ro_data = odom_dict

        radar_idx = 450
        radar_cart_list , origin_radar= data_utils.load_radar_data(sequence_path, ro_data, radar_idx, history_scans_num, cart_resolution)

        new_radar_list = data_utils.crop_radar(radar_cart_list, (256, 256))
        final_output = np.zeros_like(radar_cart_list[0])
        origin_radar_out = np.zeros_like(new_radar_list[0])
        for i in range(len(radar_cart_list)):
            final_output += radar_cart_list[i]
            origin_radar_out += new_radar_list[i]
        str_format = '{:06d}'
        radar_filename = os.path.join(sequence_path, 'Navtech_Polar', str_format.format(6) + '.png')
        cart = data_utils.polar_to_cart(cv2.imread(radar_filename,1), cart_resolution, 512, True)
        cv2.imshow("cart", cart)
        cv2.imshow("agg", final_output)
        cv2.imshow("agg_origin", origin_radar_out)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
    pass