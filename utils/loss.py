import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.optim import lr_scheduler
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = {
    'motion_weight': [1, 0.1],
    'disp_weight': [1, 0.05],

}


class MotionLoss(nn.Module):
    """
    Calculate the state estimation loss

    Parameters
    ----------
    use_weighted_loss: whether to use the weighted loss
    motion_gt:  ground_truth of the moving object 
    motion_pred: state prediction result ,torch.Size([bs, 2, 256, 256])
    config['motion_weight']: [moving, static] use to solve the data imbalance problem during training 
    """
    def __init__(self, use_weighted_loss):
        super(MotionLoss, self).__init__()
        self.use_weighted_loss = use_weighted_loss
       
    def forward(self, motion_gt, motion_pred, mask):
        motion_gt = motion_gt.view(-1, 512, 512, 2)
        motion_gt_numpy = motion_gt.numpy()
        motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
        
        log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

        valid_mask = mask[:, 0] # torch.Size([bs, 256, 256])
        valid_mask = valid_mask.to(device)

        if self.use_weighted_loss:
            motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1 # [bs, 256, 256]
            motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
            for k in range(len(config['motion_weight'])):
                weight_mask = motion_gt_numpy == (k + 1)
                #print("w",weight_mask.shape)
                motion_weight_map[weight_mask] = config['motion_weight'][k]
                
            motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
            loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
        else:
            
            loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) # torch.size([bs, 256, 256]) 
        #print(loss_speed.shape)
        loss_motion = torch.sum(loss_speed * valid_mask) / torch.nonzero(valid_mask).size(0)
        #loss_motion = torch.sum(loss_speed) / (loss_speed.shape[0] * loss_speed.shape[1] * loss_speed.shape[2])
      
        return  loss_motion

    

class ClassLoss(nn.Module):
    """
    Calculate the class estimation loss

    Parameters
    ----------
    use_weighted_loss: whether to use the weighted loss
    pixel_radar_map_gt:  ground_truth of the class, here use the radar masked by lidar result
    class_pred: class prediction result  
    raw_radars: [moving, static] use to solve the data imbalance problem during training 
    """
    def __init__(self, use_weighted_loss) -> None:
        super(ClassLoss, self).__init__()
        self.use_weighted_loss = use_weighted_loss

    def forward(self, pixel_radar_map_gt, class_pred, raw_radars):
        log_softmax_probs = F.log_softmax(class_pred, dim=1)
        if self.use_weighted_loss:
            power_weight_map = torch.clone(raw_radars[:,0,:,:,0])
            power_weight_mean = torch.mean(power_weight_map,dim=(1,2))
            power_weight_mean = torch.unsqueeze(torch.unsqueeze(power_weight_mean,1),2)

            # print("pixel_radar", pixel_radar_map_gt.shape)
            # print("log_soft_max", log_softmax_probs.shape)
            # print("power_map", power_weight_map.shape)
            # print("power_mean", power_weight_mean.shape)
            # print(torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1).shape)
            #loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1) * power_weight_map / power_weight_mean 
            loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
            # loss_class = torch.sum(- pixel_radar_map_gt_thres * log_softmax_probs, dim=1) # use thres gt and no weight
        else:
            loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
        loss_class = torch.sum(loss_class) / (class_pred.shape[2]*class_pred.shape[3])
        
        return loss_class
        

class DispLoss(nn.Module):
    def __init__(self, use_weighted_loss) -> None:
        super(DispLoss, self).__init__()
        self.use_weighted_loss = use_weighted_loss

    def forward(self, all_disp_field_gt, future_frames_num, pixel_lidar_map_gt, disp_pred, criterion):

        all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2) # [1, future_num*bs, 256, 256, 2]
        gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous() # [1, future_num*bs, 256, 256, 2]
        gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4)) # [future_num*bs, 256, 256, 2]
        gt = gt.permute(0, 3, 1, 2).to(device) # [future_num*bs, 2, 256, 256]

        valid_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256))) # [future_num*bs, 1, 256, 256] # !!! only work when future_num = 1 
        valid_mask[:,0] = pixel_lidar_map_gt[:,0] # pixel_radar_map_gt # torch.Size([bs, 256, 256])
        
        #valid_mask = valid_mask * torch.logical_not(motion_mask) 
        valid_mask = valid_mask.to(device)

        #pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num) 
        if self.use_weighted_loss:
            loss_disp = criterion(gt*valid_mask, disp_pred*valid_mask)
            axis_weight_map = torch.zeros_like(loss_disp, dtype=torch.float32)
            axis_weight_map[:,0,:,:] = 1 # right
            axis_weight_map[:,1,:,:] = 0.5 # 1 # top
            if torch.nonzero(valid_mask).size(0) != 0:
                loss_disp = torch.sum(loss_disp * axis_weight_map) / torch.nonzero(valid_mask).size(0)
                #loss_disp_value = loss_disp.item()
                #print('loss_disp', loss_disp)
            else:
                assert("The result is wrong")
        else:
            loss_disp = criterion(gt*valid_mask, disp_pred*valid_mask) / torch.nonzero(valid_mask).size(0)               
        
        return loss_disp

class CalculateLoss(nn.Module):
    def __init__(self, data, disp_pred, motion_pred, class_pred, criterion, hparams) -> None:
        super(CalculateLoss, self).__init__()
        self.raw_radars, self.pixel_radar_map_gt, self.pixel_lidar_map_gt, self.motion_gt, self.all_disp_field_gt = data
        self.motion_pred = motion_pred
        self.class_pred = class_pred
        self.disp_pred = disp_pred
        self.criterion = criterion
        self.hparams = hparams

    def forward(self):
        
        self.pixel_radar_map_gt = self.pixel_radar_map_gt.view(-1, 256, 256, self.hparams.cell_category_num)
        self.pixel_radar_map_gt = self.pixel_radar_map_gt.permute(0, 3, 1, 2).to(device)
        
        self.raw_radars = self.raw_radars.view(-1, self.hparams.num_past_frames, 256, 256, self.hparams.height_feat_size)
        self.raw_radars = self.raw_radars.to(device)
        if self.hparams.use_motion_loss:
            motion_gt = self.motion_gt.view(-1, 256, 256, 2)
            motion_gt_numpy = motion_gt.cpu().numpy()
            motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
            
            log_softmax_motion_pred = F.log_softmax(self.motion_pred, dim=1)

            valid_mask = self.pixel_radar_map_gt[:, 0] # torch.Size([bs, 256, 256])
            #print(self.pixel_radar_map_gt[:, 0].shape)
            valid_mask = valid_mask.to(device)
            # print(motion_gt.dtype)
            # print(log_softmax_motion_pred.dtype)
            # print(valid_mask.dtype)
            if self.hparams.use_weighted_loss:
                
                motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
                motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
                for k in range(len(config['motion_weight'])):
                    weight_mask = motion_gt_numpy == (k + 1)
                    motion_weight_map[weight_mask] = config['motion_weight'][k]
                    
                motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
                loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
            else:
                
                loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) # torch.size([bs, 256, 256]) 
            loss_motion = torch.sum(loss_speed * valid_mask) / torch.nonzero(valid_mask).size(0)
        else:
            loss_motion = -1



        if self.hparams.use_class_loss:
            log_softmax_probs = F.log_softmax(self.class_pred, dim=1)
            if self.hparams.use_weighted_loss:
                power_weight_map = torch.clone(self.raw_radars[:,0,:,:,0])
                power_weight_mean = torch.mean(power_weight_map,dim=(1,2))
                power_weight_mean = torch.unsqueeze(torch.unsqueeze(power_weight_mean,1),2)

                # print("pixel_radar", self.pixel_radar_map_gt.shape)
                # print("log_soft_max", log_softmax_probs.shape)
                # print("power_map", power_weight_map.shape)
                # print("power_mean", power_weight_mean.shape)
                # print(torch.sum(- self.pixel_radar_map_gt * log_softmax_probs, dim=1).shape)

                # pixel_radar torch.Size([4, 2, 256, 256])
                # log_soft_max torch.Size([4, 2, 256, 256])
                # power_map torch.Size([4, 256, 256])
                # power_mean torch.Size([4, 1, 1])
                # torch.Size([4, 256, 256])
                loss_class = torch.sum(- self.pixel_radar_map_gt * log_softmax_probs, dim=1) * power_weight_map / power_weight_mean 
                # loss_class = torch.sum(- pixel_radar_map_gt_thres * log_softmax_probs, dim=1) # use thres gt and no weight
            else:
                loss_class = torch.sum(- self.pixel_radar_map_gt * log_softmax_probs, dim=1)
            loss_class = torch.sum(loss_class) / (self.class_pred.shape[2] * self.class_pred.shape[3])
        
        else:
            loss_class = -1


        if self.hparams.use_disp_loss:

            self.all_disp_field_gt = self.all_disp_field_gt.view(-1, self.hparams.num_future_frames, 256, 256, 2) # [1, future_num*bs, 256, 256, 2]
            gt = self.all_disp_field_gt[:, -self.hparams.num_future_frames:, ...].contiguous() # [1, future_num*bs, 256, 256, 2]
            gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4)) # [future_num*bs, 256, 256, 2]
            gt = gt.permute(0, 3, 1, 2).to(device) # [future_num*bs, 2, 256, 256]

            valid_mask = torch.from_numpy(np.zeros((gt.shape[0], 1, 256, 256))) # [future_num*bs, 1, 256, 256] # !!! only work when future_num = 1 
            valid_mask[:,0] = self.pixel_lidar_map_gt[:,0] # pixel_radar_map_gt # torch.Size([bs, 256, 256])
            
            #valid_mask = valid_mask * torch.logical_not(motion_mask) 
            valid_mask = valid_mask.to(device)

            #pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num) 
            if self.hparams.use_weighted_loss:
                loss_disp = self.criterion(gt*valid_mask, self.disp_pred*valid_mask)
                axis_weight_map = torch.zeros_like(loss_disp, dtype=torch.float32)
                axis_weight_map[:,0,:,:] = 1 # right
                axis_weight_map[:,1,:,:] = 0.5 # 1 # top
                if torch.nonzero(valid_mask).size(0) != 0:
                    loss_disp = torch.sum(loss_disp * axis_weight_map) / torch.nonzero(valid_mask).size(0)
                    #loss_disp_value = loss_disp.item()
                    #print('loss_disp', loss_disp)
                else:
                    assert("The result is wrong")
            else:
                loss_disp = self.criterion(gt*valid_mask, self.hparams.disp_pred*valid_mask) / torch.nonzero(valid_mask).size(0)               

        if self.hparams.use_class_loss and self.hparams.use_disp_loss and self.hparams.use_motion_loss:
            loss = loss_class + loss_motion + loss_disp
        elif self.hparams.use_class_loss and self.hparams.use_disp_loss and (not self.hparams.use_motion_loss):
            loss = loss_class + loss_disp
        elif self.hparams.use_class_loss and (not self.hparams.use_disp_loss) and self.hparams.use_motion_loss:
            loss = loss_class + loss_motion
        elif (not self.hparams.use_class_loss) and self.hparams.use_disp_loss and (not self.hparams.use_motion_loss):
            loss = loss_disp
        elif self.hparams.use_class_loss and (not self.hparams.use_disp_loss) and (not self.hparams.use_motion_loss):
            loss = loss_class
        else:
            loss = 0

        return loss, loss_class, loss_motion, loss_disp

class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler
        
        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min
        
        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()
        
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0    # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:            
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 *\
                            (np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1)
            
        self.cur += 1
        
        return [lr for base_lr in self.base_lrs]
