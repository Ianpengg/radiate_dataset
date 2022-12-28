from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from multiprocessing import Manager
# from data.data_utils import classify_speed_level
import cv2


class RadiateDataset(Dataset):
    def __init__(self, dataset_root=None, spatial_val_num=None, future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]),  num_past_frames=2,
                 num_future_frames=1, num_category=2, cache_size=10000):
        

        
        if dataset_root is None:
            raise ValueError("The dataset root is None. Should specify its value.")
        self.dataset_root = dataset_root
        print("spatial_val_num", spatial_val_num)
        if spatial_val_num != -1:
          data = self.dataset_root+'/city_1_1'
          seq1_dirs = [os.path.join(data, f)for f in os.listdir(data)
                           if os.path.isfile(os.path.join(data, f))]
          data = self.dataset_root+'/city_2_0'
          seq2_dirs = [os.path.join(data, f)for f in os.listdir(data)
                           if os.path.isfile(os.path.join(data, f))]
          data = self.dataset_root+'/city_5_0'
          seq3_dirs = [os.path.join(data, f)for f in os.listdir(data)
                           if os.path.isfile(os.path.join(data, f))]
          if spatial_val_num == 1:
            seq_dirs = seq2_dirs+seq3_dirs
          elif spatial_val_num == 2:
            seq_dirs = seq1_dirs+seq3_dirs
          elif spatial_val_num == 3:
            seq_dirs = seq1_dirs+seq2_dirs
          else:
            print("WRONG spatial_val_num !!!")
        else:
          seq_dirs = [os.path.join(self.dataset_root, f)for f in os.listdir(self.dataset_root)
                           if os.path.isfile(os.path.join(self.dataset_root, f))]

        
        
        self.seq_dirs = seq_dirs
        self.num_sample_seqs = len(self.seq_dirs)

        self.future_frame_skip = future_frame_skip
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames

        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict_list = self.cache[idx]
        else:
            seq_dir = self.seq_dirs[idx]

            gt_file_paths = [seq_dir]
            num_gt_files = 1

            gt_dict_list = list()
            for f in range(num_gt_files):  # process the files, starting from 0.npy to 1.npy, etc
                gt_file_path = gt_file_paths[f]
                gt_data_handle = np.load(gt_file_path, allow_pickle=True)
                gt_dict = gt_data_handle.item()
                gt_dict_list.append(gt_dict)

            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict_list

        raw_radars_list = list()
        pixel_moving_map_list = list()
        pixel_radar_map_list = list()

        for gt_dict in gt_dict_list:
            dims = np.array([512, 512])
            raw_radars = list()
            num_past_pcs = self.num_past_frames
            for i in range(num_past_pcs):
                raw_radars.append(np.expand_dims(gt_dict['raw_radar_' + str(i)], axis=2))
            raw_radars = np.stack(raw_radars, 0).astype(np.float32)

            pixel_radar_map_ = gt_dict['gt_radar_pixel']
            pixel_radar_map_[pixel_radar_map_ > 0] = 1
            pixel_radar_map = np.zeros((dims[0],dims[1],2))
            # ipdb.set_trace()
            pixel_radar_map[:,:,0] = pixel_radar_map_ # valid
            pixel_radar_map[:,:,1] = np.logical_not(pixel_radar_map_) # invalid
            
            
            pixel_moving_map_ = gt_dict['gt_car_mask']
            pixel_moving_map_[pixel_moving_map_ > 0] = 1
            pixel_moving_map = np.zeros((dims[0],dims[1],2))
            pixel_moving_map[:,:,0] = pixel_moving_map_ # moving
            pixel_moving_map[:,:,1] = np.logical_not(pixel_moving_map_) # static
            #print(pixel_moving_map)
            
            
            raw_radars_list.append(raw_radars)
            pixel_moving_map_list.append(pixel_moving_map)
            pixel_radar_map_list.append(pixel_radar_map)
        #raw_radars_list = np.stack(raw_radars_list, 0)
        raw_radars_list = np.squeeze(np.array(raw_radars_list), 0)
        pixel_moving_map_list = np.stack(pixel_moving_map_list, 0)
        pixel_radar_map_list = np.stack(pixel_radar_map_list, 0)

        return raw_radars_list, pixel_moving_map_list, pixel_radar_map_list

if __name__ == "__main__":
  trainset = RadiateDataset(dataset_root='/media/ee904/Data_stored/radiate/data/training/city_1_1',spatial_val_num=-1)
  loader = DataLoader(trainset, batch_size=4, shuffle=False, )
  
  for i, data in enumerate(loader):
    print(data[0].shape)
    #print([i.shape for i in data])
    exit()