import argparse
import os
import sys
import time
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import imageio
#from model import MotionNet, MotionNetMGDA, FeatEncoder
from pyquaternion import Quaternion
from shutil import copytree, copy
#from utils.data_utils import voxelize_occupy#, calc_displace_vector, point_in_hull_fast
from model.model import RaMNet

global frame_idx
frame_idx = 5

global class_error_sum, motion_error_sum, count
class_error_sum=0.
motion_error_sum=0.
count=0.



def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
  idx = int(idx)
  return idx



def viz_combined(img, denoised_img, motion_seg):
  viz_img = np.zeros((512,512,3))
  viz_img = np.stack((img,img,img), axis=2)
  viz_denoised_img = np.zeros((512,512,3))
  if plot_motion_seg:
    viz_denoised_img[:,:,2] = (denoised_img * np.logical_not(motion_seg))
    viz_seg = np.zeros((512,512,3))
    viz_seg[:,:,0] = motion_seg
    return (viz_img*2+viz_denoised_img+viz_seg)/2.
  else:
    viz_denoised_img[:,:,2] = denoised_img
    return (viz_img*2+viz_denoised_img)/2.
# def viz_combined(img, denoised_img, motion_seg):
#   viz_img = np.zeros((256,256,3))
#   viz_img = np.stack((img,img,img), axis=2)
#   viz_denoised_img = np.zeros((256,256,3))
#   if plot_motion_seg:
#     viz_denoised_img[:,:,2] = (denoised_img * np.logical_not(motion_seg))
#     viz_seg = np.zeros((256,256,3))
#     viz_seg[:,:,0] = motion_seg
#     return (viz_img*2+viz_denoised_img+viz_seg)/2.
#   else:
#     viz_denoised_img[:,:,2] = denoised_img
#     return (viz_img*2+viz_denoised_img)/2.

def viz_denoise_gt(img, denoised_img, gt_img):
  viz_img = np.zeros((512,512,3))
  viz_img = np.stack((img,img,img), axis=2)

  viz_only_denoised_img = np.zeros((512,512,3))
  viz_only_denoised_img[:,:,2] = denoised_img*np.logical_not(gt_img)

  viz_only_gt_img = np.zeros((512,512,3))
  viz_only_gt_img[:,:,0] = gt_img*np.logical_not(denoised_img)

  viz_correct_img = np.zeros((512,512,3))
  viz_correct_img[:,:,1] = gt_img*denoised_img

  return (viz_img*4+viz_only_denoised_img+viz_only_gt_img+viz_correct_img)/4.



def gt_to_pixel_map_gt(radar_gt):
  pixel_radar_map = np.zeros((radar_gt.shape[0],radar_gt.shape[1],2))
  pixel_radar_map[:,:,0] = radar_gt # valid
  pixel_radar_map[:,:,1] = np.logical_not(radar_gt) # invalid
  pixel_radar_map_list = list()
  pixel_radar_map_list.append(pixel_radar_map)
  pixel_radar_map_list = np.stack(pixel_radar_map_list, 0)
  pixel_radar_map_list = torch.from_numpy(pixel_radar_map_list)
  pixel_radar_map_gt = pixel_radar_map_list.permute(0, 3, 1, 2)
  return pixel_radar_map_gt



def plot_quiver(ax_, disp0, disp1, viz_cat, viz_motion):
  # Plot quiver.
  field_gt = np.zeros((256,256,2))
  field_gt[:,:,0] = disp0
  field_gt[:,:,1] = disp1
  idx_x = np.arange(field_gt.shape[0])
  idx_y = np.arange(field_gt.shape[1])
  idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
  # For cells with very small movements, we threshold them to be static
  field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
  thd_mask = field_gt_norm <= 0.5
  field_gt[thd_mask, :] = 0
  # Get the displacement field
  mask = viz_cat.astype(np.bool) * viz_motion.astype(np.bool)
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  qk1 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=15, headaxislength=15, color='r', alpha=0.9, minlength=6, minshaft=1) #'g'
#   qk1 = ax_.quiver(Y, X, U, V, angles="xy", scale_units='xy', scale=0.35, color='r') 
  mask = viz_cat.astype(np.bool) * np.logical_not(viz_motion.astype(np.bool))
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  #qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=10, headaxislength=10, color='mediumblue', alpha=0.9, minlength=6, minshaft=1)
  qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.35, width=0.0005, headwidth=30, headlength=10, headaxislength=10, color=[(0.1,0.15,1.0)], alpha=0.9, minlength=6, minshaft=1)
#   qk2 = ax_.quiver(Y, X, U, V, angles="xy", scale_units='xy', scale=0.35, color=[(0.1,0.15,1.0)]) 
  # dodgerblue
  return qk1, qk2


def flow_to_img(flow_x, flow_y):
  hsv = np.zeros((flow_x.shape[0],flow_x.shape[1],3)).astype(np.uint8)
  hsv[...,1] = 255
  mag, ang = cv2.cartToPolar(flow_x, flow_y)
  mag[mag>=15]=15 # upper bound
  hsv[...,0] = ang*180/np.pi/2
  #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  hsv[...,2] = mag/15. * 255
  rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  return rgb.astype(np.float32)/255.

def opticalflow(prvs_,next_):
    hsv = np.zeros((prvs_.shape[0],prvs_.shape[1],3)).astype(np.uint8)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prvs_,next_, None, 0.5, 3, 12, 3, 5, 1.2, 0) # 6
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb.astype(np.float32)/255.


def gen_corr_line_set(src, dst, corres, color):
    viz_points = np.concatenate((src, dst), axis=1)
    viz_lines = list()
    for corr in corres:
      associate_shift = corr[1]-corr[0]
      viz_lines.append([corr[0],corr[0]+src.shape[1]])
    colors = [color for i in range(len(viz_lines))]
    line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(viz_points.T),
            lines=o3d.utility.Vector2iVector(viz_lines),
        )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def calc_odom_by_disp_map(disp0, disp1, radar_mask, moving_mask):
  disp_map = np.zeros((256,256,2))
  disp_map[:,:,0] = disp0
  disp_map[:,:,1] = disp1

  radar_mask[radar_mask>0]=1
  radar_mask[radar_mask<=0]=0

  pointcloud = list()
  pointcloud_ = list()
  N=256
  center=N/2-0.5
  for i in range(N): # row
    for j in range(N): # col
      if radar_mask[i,j]==1:
        point = np.array([center-i, j-center, 0]) # x, y in ego-motion frame
        delta = np.array([disp_map[i,j,1], disp_map[i,j,0], 0])
        point_ = point + delta
        pointcloud.append(point)
        pointcloud_.append(point_)
  pc = np.array(pointcloud)
  pc_ = np.array(pointcloud_)
  #print(pc.shape)
  #print(pc_.shape)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc)
  pcd.paint_uniform_color([1, 0, 0])
  pcd_ = o3d.geometry.PointCloud()
  pcd_.points = o3d.utility.Vector3dVector(pc_)
  pcd_.paint_uniform_color([0, 1, 0])
  arr = np.expand_dims(np.arange(pc.shape[0]),axis=0)
  np_corres = np.concatenate((arr, arr), axis=0).T
  corres = o3d.utility.Vector2iVector(np_corres)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  M, mask = cv2.findHomography(pc, pc_, cv2.RANSAC, 3.0) # 1~10 => strict~loose 2

  # gen outlier mask
  outlier_mask = np.copy(radar_mask).astype(np.bool)
  pc_inlier = np.delete(pc, np.where(mask==0), axis=0)
  for point in pc_inlier:
    i = (center-point[0]).astype(np.int)
    j = (center+point[1]).astype(np.int)
    outlier_mask[i,j] = 0

  #print(M[1,2], M[0,2], -np.arctan(M[1,0]/M[0,0])) # w.r.t. disp frame
  np_corres_new = np_corres[mask.squeeze().astype(np.bool),:]
  corres_new = o3d.utility.Vector2iVector(np_corres_new)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres_new, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  return M, outlier_mask


def vis_result(data_path, trained_model_path, img_save_dir,  which_model, disp, save):
    global frame_idx
    global class_error_sum, motion_error_sum, count
    fig1 = None
    fig2 = None
    data_dirs = [os.path.join(data_path, f)for f in os.listdir(data_path)
                    if os.path.isfile(os.path.join(data_path, f))]
    data_dirs.sort(key=file_to_id)
    print(len(data_dirs))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pre-trained network weights
    loaded_models = list()
    if which_model == "RaMNet":
        model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
        
        #model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'], False)
        model = model.to(device)

        loaded_models = [model]

    else:
        model = MotionNet(out_seq_len=20, motion_category_num=5, height_feat_size=13)

        #model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        #print(model.summary())
        print(checkpoint['model_state_dict'].keys())
        model.load_state_dict(checkpoint['model_state_dict'], False)
        model = model.to(device)

        loaded_models = [model] 
        print('model error')
    print("Loaded pretrained model {}".format(which_model))

    for data in data_dirs:
        count += 1
        if count < 0: # 800, 600 -> moving obj # 540 ->left-right moving obj
            continue
        print('---------')
        print(data)
        radar_idx = file_to_id(data)

        raw_radars = []
        
        gt_data_handle = np.load(data, allow_pickle=True)
        gt_dict = gt_data_handle.item()

        num_past_pcs = num_past_frames
        #num_past_pcs = 2
        for i in range(num_past_pcs):
            
            #raw_radars.append(np.expand_dims(gt_dict['raw_radar_' + str(i)][256-128:256+128, 256-128:256+128], axis=2))
            raw_radars.append(np.expand_dims(gt_dict['raw_radar_' + str(i)], axis=2))
        raw_radars = np.stack(raw_radars, 0).astype(np.float32)
        raw_radars_list = []
        raw_radars_list.append(raw_radars)
        raw_radars = np.stack(raw_radars_list, 0)
        raw_radars = torch.tensor(raw_radars).to(device)
        #raw_radars_input = torch.tensor(raw_radars_list).to(device)
        raw_radar = raw_radars[0,0,:,:,0]
        #print(raw_radars_input.shape)

        # raw_radars = raw_radars_list.view(-1, num_past_frames, 512, 512, height_feat_size)
        # raw_radars = raw_radars.to(device)
        #ipdb.set_trace()
        #raw_radar = raw_radars[0,0,:,:,0]
        # print(raw_radars)
        # print(raw_radar)
        radar_gt = gt_dict['gt_radar_pixel']
        radar_gt[radar_gt > 0] = 1
        motion_gt = gt_dict['gt_car_mask']
        motion_gt[motion_gt > 0] = 1

        

        # model = loaded_models[0]
        model.eval()

        with torch.no_grad():
            # network estimate #
            if use_temporal_info:
            
                disp_pred, cat_pred, motion_pred = model(raw_radars)
            # else:
            #     disp_pred, cat_pred, motion_pred = model(raw_radars_curr)
        #print('disp_pred:',disp_pred.shape)

        # compute class error #
        # log_softmax_probs = F.log_softmax(cat_pred, dim=1) # torch.Size([1, 2, 256, 256])
        # #print(log_softmax_probs[0,:,:3,:3])
        # pixel_radar_map_gt = gt_to_pixel_map_gt(radar_gt).to(device) # torch.Size([1, 2, 256, 256])
        # loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
        # loss_class = torch.sum(loss_class) / (cat_pred.shape[2]*cat_pred.shape[3])
        # print('class loss:', loss_class.item())
        # class_error_sum += loss_class.item()
        # print('count:', count, 'avg class loss:', class_error_sum/count)

        # # # compute motion error #
        # log_softmax_probs = F.log_softmax(motion_pred, dim=1) # torch.Size([1, 2, 256, 256])
        # pixel_radar_map_gt = gt_to_pixel_map_gt(motion_gt).to(device) # torch.Size([1, 2, 256, 256])
        # loss_motion = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
        # loss_motion = torch.sum(loss_motion) / (motion_pred.shape[2]*motion_pred.shape[3])
        # print('motion loss:', loss_motion.item())
        # motion_error_sum += loss_motion.item()
        # print('count:', count, 'avg motion loss:', motion_error_sum/count)

        if save or disp:

       
            # convert all output to numpy
            cat_pred_numpy = cat_pred.cpu().numpy()
            motion_pred_numpy = motion_pred.cpu().numpy()
            disp_pred_numpy = disp_pred.cpu().numpy()
            raw_radars = raw_radars.cpu().numpy()
            #raw_radars = raw_radars.detach().numpy()
            raw_radar = raw_radar.cpu().detach().numpy()
            # class_pred_viz = (np.zeros((cat_pred_numpy.shape[0],3,cat_pred_numpy.shape[2],cat_pred_numpy.shape[3])))
            # visualize network output #
            viz_cat_pred = cat_pred_numpy[0,:,:,:].argmin(axis=0)
            # class_pred_viz[:,0,:,:] = cat_pred_numpy[:,0,:,:]>0.5
            # class_pred_viz[:,1,:,:] = cat_pred_numpy[:,0,:,:]>0.5
            # class_pred_viz[:,2,:,:] = cat_pred_numpy[:,0,:,:]>0.5
            # viz_cat_pred = class_pred_viz[0,0,:,:]
            #cv2.imshow("test", viz_cat_pred)
            #print(viz_cat_pred)
            #cv2.waitKey()
            #ipdb.set_trace()
            #print(motion_pred_numpy.shape)
            viz_motion_pred = motion_pred_numpy[0,0,:,:]
            #print(viz_motion_pred)
            viz_motion_pred[viz_motion_pred > 0] = 1
            viz_motion_pred[viz_motion_pred <= 0] = 0
            #print(cat_pred_numpy)
            viz_motion_pred = viz_motion_pred  #* viz_cat_pred


            cv2.imshow("mask", viz_motion_pred)
            #cv2.imshow("test", raw_radar/255.0)
            #print(raw_radar)
            
            if args.disp_seg:
                if fig1 == None:
                    fig1, ax1 = plt.subplots(1, 2, figsize=(32, 32))
                print(raw_radar.shape, radar_gt.shape, motion_gt.shape)
                ax1[0].imshow(np.clip(viz_combined(raw_radar / 255., radar_gt, motion_gt), 0, 1.0))  
                ax1[0].axis('off')
                ax1[0].set_aspect('equal')
                ax1[0].title.set_text('GT')
                
                #ax1[0].imshow(viz_combined(raw_radar, radar_gt, motion_gt))
                test = np.zeros_like(raw_radar)
                #print(np.clip(viz_combined(raw_radar/255. , viz_cat_pred, viz_motion_pred), 0, 1.0))
                test = np.zeros_like(viz_cat_pred)
                ax1[1].imshow(np.clip(viz_combined(raw_radar/255., test, viz_motion_pred), 0, 1.0))
                ax1[1].axis('off')
                ax1[1].set_aspect('equal')
                ax1[1].title.set_text('Result')
                # plt.show()
                #ipdb.set_trace()    
            if args.disp_quiver:
                if fig2 == None:
                    fig2, ax2 = plt.subplots(1, 2, figsize=(21,9))
                ax2[0].imshow(viz_combined(raw_radar, radar_gt, motion_gt))
                #ax2[0].imshow(raw_radar, cmap="gray")
                ax2[0].axis('off')
                ax2[0].set_aspect('equal')
                ax2[0].title.set_text('GT')

                ax2[1].imshow(viz_combined(raw_radar, viz_cat_pred, viz_motion_pred))
                ax2[1].axis('off')
                ax2[1].set_aspect('equal')
                ax2[1].title.set_text('Result')
                gt_qk1, gt_qk2 = plot_quiver(ax2[0], -disp0_gt[:,:], disp1_gt[:,:], lidar_gt, motion_gt)

                qk1, qk2 = plot_quiver(ax2[1], -disp_pred_numpy[0,0], disp_pred_numpy[0,1], radar_gt, viz_motion_pred)
                if save:
                    fig2.savefig(os.path.join(img_save_dir, str(radar_idx) + '.png'), bbox_inches='tight')
                    plt.close()
        
            
                #plt.close()
            #plt.show()
            plt.pause(0.1)
            if args.disp_seg:
                ax1[0].clear()
                ax1[1].clear()
            if args.disp_quiver:
                ax2[0].clear()
                ax2[1].clear()
                gt_qk1.remove()
                gt_qk2.remove()
                qk1.remove()
                qk2.remove()

            if save or disp:
                frame_idx = frame_idx + 1

def frame_to_gif(img_save_dir):
    images = np.sort([im for im in os.listdir(img_save_dir) if os.path.isfile(os.path.join(img_save_dir, im))
              and im.endswith('.png')])

    #print(images)
    num_images = len(images)
    save_gif_path = os.path.join(img_save_dir, 'result.gif')
    with imageio.get_writer(save_gif_path, mode='I', fps=5) as writer:
        for i in range(num_images):
            image_file = os.path.join(img_save_dir, images[i])
            image = imageio.imread(image_file)
            writer.append_data(image)

            print("Appending image {}".format(i))

color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}


num_past_frames = 2
out_seq_len = 1
height_feat_size=1
plot_motion_seg = True
use_temporal_info = True


parser = argparse.ArgumentParser()
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='figs/', help='The path to the output log file')
parser.add_argument('--modelpath', default=None, type=str, help='Path to the pretrained model')
parser.add_argument('-d', '--datapath', default=None, type=str, help='The path to Oxford ground-truth data')
parser.add_argument('--net', default='RaMNet', type=str, help='Which network [MotionNet/MotionNetMGDA]')
parser.add_argument('--disp', action='store_true', help='Whether to immediately show the images')
parser.add_argument('--save', action='store_true', help='Whether to save the images')
parser.add_argument('--disp_seg', action='store_true', help='Whether to show the segmentation result')
parser.add_argument('--disp_quiver', action='store_true', help='Whether to show the quiver result')
parser.add_argument('--video', action='store_true', help='Whether to generate images or [gif/mp4]')

args = parser.parse_args()

gen_prediction_frames = True
if_disp = args.disp
if_save = args.save
#image_save_dir = check_folder(args.savepath)

if if_save:
    matplotlib.use('Agg')

    
# if gen_prediction_frames:
#     vis_result(data_path=args.data, trained_model_path=args.modelpath, which_model=args.net, disp=if_disp, save=if_save, ax=ax)
# else:
#     #frames_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
#     #save_gif_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
#     #gen_scene_prediction_video(args.savepath, args.savepath, out_format='gif')
#     pass
#args = parser.parse_args()


need_log = args.log

def main():
    
    if need_log:
        
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        image_save_path = check_folder(logger_root)
        image_save_path = check_folder(os.path.join(image_save_path, time_stamp))
        log_file_name = os.path.join(image_save_path, 'log.txt')
        saver = open(log_file_name, "w")
        saver.write("command")
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
    image_save_path = None
    #copy(args.modelpath, image_save_path)

    #saver.flush()
    #saver.close()
    #img_save_dir = image_save_path
    vis_result(data_path=args.datapath, trained_model_path=args.modelpath, img_save_dir=image_save_path, which_model=args.net, disp=if_disp, save=if_save)


   
    


if __name__ == "__main__":
    main()
    #frame_to_gif(img_save_dir='/media/ee904/Data_stored/temp/RadarMotionNet/figs/2022-10-24_11-24-02')


