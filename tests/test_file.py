import unittest
import os 
import cv2
import numpy as np
import sys
sys.path.insert(0, '/media/ee904/Data_stored/temp_i/radiate_dataset/')
import utils.data_utils as data_utils

class TestFile(unittest.TestCase):
    
    def test_file(self):
        data_root = "/data/training/city_1_1/"
        datas = np.load(os.getcwd() + data_root + '6.npy', allow_pickle=True)
        datas = datas.item()
        for i in datas:
            print(datas[i].shape)
            cv2.imshow("test", datas[i])
            cv2.waitKey()
if __name__ == "__main__":
    unittest.main()