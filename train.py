

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
import sys
import os
import math
from shutil import copytree, copy
from model.model import RaMNet
from DataLoader import RadiateDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision

from torch.utils.data import random_split
from tqdm import tqdm
import cv2
import open3d as o3d
import ipdb
import matplotlib.pyplot as plt
from utils.loss import MotionLoss, ClassLoss, DispLoss, WarmupCosineLR

config = {
'batch_size': 8,
'num_epochs': 10,
'num_workers': 4,
'use_temporal_info': True,
'num_past_frames': 2,
'future_frames_num': 1,
'out_seq_len': 1,
'height_feat_size': 1,
'cell_category_num': 2,
'motion_weight': [1.0, 0.500],
}

BATCH_SIZE = config['batch_size']
EPOCHS = config['num_epochs']
WORKERS = config['num_workers']



#import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


use_weighted_loss = True

use_class_loss = False # True
use_motion_loss = True # True
use_disp_loss = False # True

use_odom_loss = False
use_odom_net = False

use_temporal_info = True
num_past_frames = 2
out_seq_len = 1  # The number of future frames we are going to predict

val_percent = 0.05

# static
height_feat_size = 1 #13  # The size along the height dimension
cell_category_num = 2  # The number of object categories (including the background)
# no use
pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
trans_matrix_idx = 1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)

global global_step
global_step = 0

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--pretrain', default='', type=str, help='The path to the saved model that is loaded as pretrained model')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=45, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')

parser.add_argument('--nn_sampling', action='store_true', help='Whether to use nearest neighbor sampling in bg_tc loss')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')
parser.add_argument('--board', action='store_true', help='Whether to show in tensorboard')
parser.add_argument('-sv', '--spatial_val_num', default=-1, type=int, help='Section number for Spatial vaidation')
#parser.add_argument('')
args = parser.parse_args()
print(args)
# take in args

need_log = args.log
need_board = args.board

BATCH_SIZE = args.batch
num_epochs = args.nepoch
num_workers = args.nworker

use_nn_sampling = args.nn_sampling




def main():
    start_epoch = 1

    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_multi_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.write("use_class_loss: {}\n use_motion_loss: {}\n use_disp_loss: {}\n use_odom_loss: {}\n use_odom_net: {}\n use_temporal_info: {}\n use_weighted_loss: {}\n".format(use_class_loss,use_motion_loss,use_disp_loss,use_odom_loss,use_odom_net,use_temporal_info,use_weighted_loss))
            saver.flush()

            # Copy the code files as logs
            #copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
            #copytree('data', os.path.join(model_save_path, 'data'))
            copytree('utils', os.path.join(model_save_path, 'utils'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"
            #model_save_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-03-28_21-32-13_as_pretrain'

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")

            saver.flush()

        #if arg.pretrain != '':


    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
 #   area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

    trainset = RadiateDataset(dataset_root=args.data, spatial_val_num=args.spatial_val_num, future_frame_skip=0, num_past_frames=num_past_frames, num_future_frames=1, voxel_size=voxel_size,
                                    area_extents=area_extents, num_category=cell_category_num)
    n_val = int(len(trainset) * val_percent)
    n_train = len(trainset) - n_val
    train_set, val_set = random_split(trainset, [n_train, n_val])

    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print(BATCH_SIZE)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    #print(type(trainloader))
    print("Training dataset size:", len(trainset))

    model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames, use_odom_net=use_odom_net)
    
   

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.0016)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5) # 10, 20,30,40
    
    
    warm_up = 5

    #scheduler = WarmupCosineLR(optimizer, 1e-6, 0.002, warm_up, num_epochs, 0.1)
    


    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    if args.pretrain != '':
        print("load pretrain")
        checkpoint = torch.load(args.pretrain)
        start_epoch = 1

        #print('checkpoint[model_state_dict]', checkpoint['model_state_dict'] )
        model.load_state_dict(checkpoint['model_state_dict'], False) # strict=False
        #print(len(list(model.parameters())))
        if isinstance(model,torch.nn.DataParallel):
          model = model.module
        
        # print(model.state_classify)
        # for name, para in model.named_parameters():
        #     if name in ['module.state_classify.conv1.weight',
        #     'module.state_classify.conv1.bias', 
        #     'module.state_classify.conv2.weight',
        #     'module.state_classify.conv2.bias',
        #     'module.state_classify.bn1.weight',
        #     'module.state_classify.bn1.bias']:
        #         para.requires_grad = True
        #     else:
        #         para.requires_grad=False 
        # optimizer = optim.Adam([{'params': model.state_classify.parameters(), 'lr': 0.0008}], lr=0.0008)
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': 0.001}],lr=0.001, momentum=0.9, weight_decay=1e-4)
        #scheduler = 
        
        print("Load model from {}, at epoch {}".format(args.pretrain, checkpoint['epoch']))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5) # 10, 20,30,40
    #model = nn.DataParallel(model)
    model = model.to(device)
    if need_board:
      writer = SummaryWriter()
    else:
      writer=-1

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        model.train()

        loss_disp, loss_class, loss_motion, disp_pred, class_pred, motion_pred \
            = train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size)
        scheduler.step()

        loss_disp_val, loss_class_val, loss_motion_val = eval(model, criterion, valloader, device, epoch, writer, voxel_size)
        if need_board:
          if use_class_loss:
            writer.add_scalar('loss_class/val', loss_class_val, epoch)
          if use_motion_loss:
            writer.add_scalar('loss_motion/val', loss_motion_val, epoch)
          if use_disp_loss:
            writer.add_scalar('loss_disp/val', loss_disp_val, epoch)
          

        if need_log:
            
            saver.write("{}\t{}\t{}\n".format(loss_disp, loss_class, loss_motion))
            saver.flush()

        # save model
        if need_log and (epoch % 1 == 0 or epoch == num_epochs or epoch == 1 or epoch > 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'loss': loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_board:
        writer.close()
    if need_log:
        saver.close()

def eval(model, criterion, valloader, device, epoch, writer, voxel_size):
  n_val = len(valloader)
  model.eval()
  loss_class_tot=0
  loss_motion_tot=0
  loss_disp_tot=0
  
  with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for i, data in enumerate(valloader, 0):
      raw_radars, pixel_moving_map_gt, pixel_radar_map_gt = data
      # Move to GPU/CPU
      raw_radars = raw_radars.view(-1, num_past_frames, 512, 512, height_feat_size)
      raw_radars = raw_radars.to(device)
      # Make prediction
      with torch.no_grad():
        if use_temporal_info == True:          
            disp_pred, class_pred, motion_pred = model(raw_radars)
        else:
          raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 512, 512, 1)).astype(np.float32))
          raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
          raw_radars_curr = raw_radars_curr.to(device)
          disp_pred, class_pred, motion_pred = model(raw_radars_curr)
            
      # Compute the losses
      optimizer=-1
      loss_class, loss_motion, loss_disp = \
          compute_and_bp_loss(optimizer, device, out_seq_len, pixel_radar_map_gt, pixel_moving_map_gt,
            criterion, class_pred, motion_pred, disp_pred, raw_radars, bp_loss=False)
      loss_class_tot += loss_class
      loss_motion_tot += loss_motion
      
      if loss_disp > 0:
        loss_disp_tot += loss_disp
#      else:
#        n_val = n_val-1

      pbar.update()
  model.train()
  return loss_disp_tot/n_val, loss_class_tot/n_val, loss_motion_tot/n_val,

def train(model, criterion, trainloader, optimizer, device, epoch, writer, voxel_size):

    running_loss_disp = AverageMeter('Disp', ':.6f')  # for motion prediction error
    running_loss_class = AverageMeter('Class', ':.6f')  # for cell classification error
    running_loss_motion = AverageMeter('Motion', ':.6f')  # for state estimation error
    
    loop = tqdm((trainloader), total = len(trainloader))
    for data in loop:
        global global_step
        global_step += 1
        
        
        raw_radars, pixel_moving_map_gt, pixel_radar_map_gt = data
#        print('---trainloader output---')
#        print(raw_radars.shape)
#        print(pixel_radar_map_gt.shape)
#        print(pixel_moving_map_gt.shape)
#        torch.Size([bs, 1, history_frame_num, 256, 256, 1])
#        torch.Size([bs, 1, 256, 256, 2])
#        torch.Size([bs, 1, 256, 256, 2])
        # print(pixel_moving_map_gt[-1,0,:,:,0].shape)
        

        # Move to GPU/CPU
        raw_radars = raw_radars.view(-1, num_past_frames, 512, 512, height_feat_size)
        raw_radars = raw_radars.to(device)
        #print(raw_radars.shape)   # torch.Size([8, 2, 256, 256, 1])

        # Make prediction
        if use_temporal_info == True:
          #print('---network input--- \n',raw_radars.shape) #input shape: torch.Size([bs*1, 5, 256, 256, 1])
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars)
            odom_pred = -1
        else:
          raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
          raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
          raw_radars_curr = raw_radars_curr.to(device)
          #print('---network input--- \n', raw_radars_curr.shape) #input shape: torch.Size([bs*1, 1, 256, 256, 1])
          if use_odom_net:
            disp_pred, class_pred, motion_pred, odom_pred = model(raw_radars_curr)
          else:
            disp_pred, class_pred, motion_pred = model(raw_radars_curr)
            odom_pred = -1
        # print('---network output---')
        # print(disp_pred.shape)
        # print(class_pred.shape)
        # print(motion_pred.shape)
        # [20*bs*2, 2, 256, 256]
        # [bs*2, 2, 256, 256]
        # [bs*2, 2, 256, 256]

        # Compute and back-propagate the losses
        loss_class, loss_motion, loss_disp = \
            compute_and_bp_loss(optimizer, device, out_seq_len, pixel_moving_map_gt, pixel_radar_map_gt,
            criterion, class_pred, motion_pred, disp_pred, raw_radars, bp_loss=True)

        if need_board:
          if loss_class>0:
            writer.add_scalar('loss_class/train', loss_class, global_step)
          if loss_motion>0:
            writer.add_scalar('loss_motion/train', loss_motion, global_step)
          if loss_disp>0:
            writer.add_scalar('loss_disp/train', loss_disp, global_step)

          if global_step%100==0:
            raw_radars_viz = torch.from_numpy(np.zeros((raw_radars.shape[0],3,raw_radars.shape[2],raw_radars.shape[3])))
            raw_radars_viz[:,0,:,:] = raw_radars[:,0,:,:,0] / 255.
            raw_radars_viz[:,1,:,:] = raw_radars[:,0,:,:,0] / 255.
            raw_radars_viz[:,2,:,:] = raw_radars[:,0,:,:,0] / 255.
            
            writer.add_images('raw_radars', raw_radars_viz, global_step)

            class_pred_viz = torch.from_numpy(np.zeros((class_pred.shape[0],3,class_pred.shape[2],class_pred.shape[3])))
            combine_class_radar_viz  = torch.from_numpy(np.zeros((class_pred.shape[0],3,class_pred.shape[2],class_pred.shape[3])))
            #print(torch.nonzero(class_pred[:,0,:,:] > 0.5))
            class_pred_viz[:,0,:,:] = class_pred[:,0,:,:]>0.5
            class_pred_viz[:,1,:,:] = class_pred[:,0,:,:]>0.5
            class_pred_viz[:,2,:,:] = class_pred[:,0,:,:]>0.5
            writer.add_images('class_pred', class_pred_viz, global_step)

            class_pred_viz[:,0,:,:] = 0
            class_pred_viz[:,1,:,:] = class_pred[:,0,:,:]>0.5
            class_pred_viz[:,2,:,:] = 0
            combine_class_radar_viz[:,:,:,:] = (raw_radars_viz *2 + class_pred_viz) / 2

            #print("class_pred.shape", class_pred.shape)
            writer.add_images('combined_class_radar',combine_class_radar_viz, global_step )
            
            motion_pred_viz = torch.from_numpy(np.zeros((motion_pred.shape[0],3,motion_pred.shape[2],motion_pred.shape[3])))
            combine_motion_radar_viz  = torch.from_numpy(np.zeros((class_pred.shape[0],3,class_pred.shape[2],class_pred.shape[3])))
           
            motion_pred_viz[:,0,:,:] = motion_pred[:,0,:,:]>0.5
            motion_pred_viz[:,1,:,:] = motion_pred[:,0,:,:]>0.5
            motion_pred_viz[:,2,:,:] = motion_pred[:,0,:,:]>0.5
            writer.add_images('motion_pred', motion_pred_viz, global_step)
            motion_pred_viz[:,0,:,:] = motion_pred[:,0,:,:]>0.5
            motion_pred_viz[:,1,:,:] = 0
            motion_pred_viz[:,2,:,:] = 0
            combine_motion_radar_viz[:,:,:,:] = (raw_radars_viz *2 + motion_pred_viz) / 2
            #print("motion_pred_viz.shape", motion_pred_viz.shape)
            writer.add_images("combined_motion_radar", combine_motion_radar_viz, global_step)

            disp_pred0_viz = torch.from_numpy(np.zeros((disp_pred.shape[0],3,disp_pred.shape[2],motion_pred.shape[3])))
            disp_pred0_viz[:,0,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred0_viz[:,1,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred0_viz[:,2,:,:] = torch.abs(disp_pred[:,0,:,:])
            disp_pred1_viz = torch.from_numpy(np.zeros((disp_pred.shape[0],3,disp_pred.shape[2],motion_pred.shape[3])))
            disp_pred1_viz[:,0,:,:] = torch.abs(disp_pred[:,1,:,:])
            disp_pred1_viz[:,1,:,:] = torch.abs(disp_pred[:,1,:,:])
            disp_pred1_viz[:,2,:,:] = torch.abs(disp_pred[:,1,:,:])
            writer.add_images('disp_pred0', disp_pred0_viz, global_step)
            writer.add_images('disp_pred1', disp_pred1_viz, global_step)
            
            # ipdb.set_trace()
       

        
        running_loss_disp.update(loss_disp)
        running_loss_class.update(loss_class)
        running_loss_motion.update(loss_motion)
        loop.set_description(f'Epoch [{num_epochs}/ {epoch}]')
        loop.set_postfix(loss_motion = running_loss_motion, loss_class= running_loss_class, )

    return running_loss_disp, running_loss_class, running_loss_motion, disp_pred, class_pred, motion_pred


# Compute and back-propagate the loss
def compute_and_bp_loss(optimizer, device, future_frames_num,   
                        motion_gt,pixel_radar_map_gt,
                        criterion, class_pred, motion_pred, disp_pred, raw_radars, bp_loss):
    if bp_loss:
        optimizer.zero_grad()

    # ---------------------------------------------------------------------
    pixel_radar_map_gt = pixel_radar_map_gt.view(-1, 512, 512, cell_category_num)
    pixel_radar_map_gt = pixel_radar_map_gt.permute(0, 3, 1, 2).to(device)
    # #print('pixel_radar_map_gt.shape:', pixel_radar_map_gt.shape) # torch.Size([bs, 2, 256, 256]) # non_empty_map
    # pixel_lidar_map_gt = pixel_lidar_map_gt.to(device) # torch.Size([bs, 1, 256, 256])
    #print('pixel_lidar_map_gt.shape:', pixel_lidar_map_gt.shape)

    ### power thres
    # power_thres_map = torch.clone(raw_radars[:,0,:,:,0])
    # power_thres_map[power_thres_map>0.08] = 1
    # power_thres_map[power_thres_map<=0.08] = 0

    # pixel_radar_map_gt_thres = torch.clone(pixel_radar_map_gt)
    # pixel_radar_map_gt_thres[:,0,:,:] = pixel_radar_map_gt_thres[:,0,:,:]*power_thres_map
    # pixel_radar_map_gt_thres[:,1,:,:] = torch.logical_not(pixel_radar_map_gt_thres[:,0,:,:])

    # pixel_lidar_map_gt_thres = torch.clone(pixel_lidar_map_gt)
    # pixel_lidar_map_gt_thres[:,0,:,:] = pixel_lidar_map_gt_thres[:,0,:,:]*power_thres_map

    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].imshow(power_thres_map[0].detach().cpu().numpy())
    # ax[1].imshow(pixel_radar_map_gt[0,0].detach().cpu().numpy())
    # #ax[2].imshow(pixel_radar_map_gt_thres[0,0].detach().cpu().numpy())
    # ax[2].imshow(pixel_lidar_map_gt_thres[0,0].detach().cpu().numpy())
    # plt.show()

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    if use_class_loss:
        class_criterion = ClassLoss(use_weighted_loss=use_weighted_loss) 
        loss_class = class_criterion(pixel_radar_map_gt, class_pred, raw_radars)
        loss_class_value = loss_class.item()
    else:
        loss_class_value = -1

    # ---------------------------------------------------------------------
    # -- Compute the motion loss
    if use_motion_loss:
        motion_criterion = MotionLoss(use_weighted_loss=use_weighted_loss)
        loss_motion = motion_criterion(motion_gt, motion_pred, pixel_radar_map_gt)
        loss_motion_value = loss_motion.item()
    else:
        loss_motion_value = -1

    # ---------------------------------------------------------------------
    # -- Compute the displacement loss
    if use_disp_loss:
        disp_criterion = DispLoss(use_weighted_loss=use_weighted_loss)
        loss_disp = disp_criterion(all_disp_field_gt, future_frames_num, pixel_lidar_map_gt, disp_pred, criterion)
        loss_disp_value = loss_disp.item()
    else:
        loss_disp_value = -1

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    if use_class_loss and use_disp_loss and use_motion_loss:
      loss = loss_class + loss_motion + loss_disp
    elif use_class_loss and use_disp_loss and (not use_motion_loss):
      loss = loss_class + loss_disp
    elif use_class_loss and (not use_disp_loss) and use_motion_loss:
      loss = 0.05*loss_class + loss_motion
      
    elif (not use_class_loss) and use_disp_loss and (not use_motion_loss):
      loss = loss_disp
    elif use_class_loss and (not use_disp_loss) and (not use_motion_loss):
      loss = loss_class
    elif (not use_class_loss) and (not use_disp_loss) and use_motion_loss:
      loss = loss_motion 
    else:
      loss = 0


    if bp_loss:
      loss.backward()
      optimizer.step()

    return loss_class_value, loss_motion_value, loss_disp_value

# We name it instance spatial-temporal consistency loss because it involves each instance

if __name__ == "__main__":
    main()