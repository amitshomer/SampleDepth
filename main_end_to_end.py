"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import numpy as np
import os
import sys
import time
import shutil
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import Models
import Datasets
import warnings
import random
import matplotlib.pyplot as plt
from Models.PredNet import PredNet

from datetime import datetime
from Models.SampleDepth import SampleDepth
from Models.Global_mask import Global_mask
from Loss.loss import define_loss, allowed_losses, MSE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import plot_images, str2bool, define_optim, define_scheduler, \
                        Logger, AverageMeter, first_run, mkdir_if_missing, \
                        define_init_weights, init_distributed_mode, sample_random

cmap = plt.cm.jet

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

# Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--dataset', type=str, default='kitti', help='dataset to work with')
parser.add_argument('--nepochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--thres', type=int, default=0, help='epoch for pretraining')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')
parser.add_argument('--mod', type=str, default='mod', choices=Models.allowed_models(), help='Model for use')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--val_batch_size', default=None, help='batch size selection validation set')
parser.add_argument('--learning_rate', metavar='lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')

parser.add_argument('--evaluate', action='store_true', help='only evaluate')
parser.add_argument('--resume', type=str, default='', help='resume latest saved run number')
parser.add_argument("--resume_bool", type=str2bool, nargs='?', const=True, default=False, help="True to start train from resume number")
parser.add_argument('--nworkers', type=int, default=8, help='num of threads')
parser.add_argument('--nworkers_val', type=int, default=8, help='num of threads')
parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
parser.add_argument('--subset', type=int, default=None, help='Take subset of train set')
parser.add_argument('--input_type', type=str, default='rgb', choices=['depth','rgb'], help='use rgb for rgbdepth')
parser.add_argument('--side_selection', type=str, default='', help='train on one specific stereo camera')
parser.add_argument('--no_tb', type=str2bool, nargs='?', const=True,
                    default=True, help="use mask_gt - mask_input as final mask for loss calculation")
parser.add_argument('--test_mode', action='store_true', help='Do not use resume')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained model')
parser.add_argument('--load_external_mod', type=str2bool, nargs='?', const=True, default=False, help='path to external mod')
parser.add_argument('--dataset_name', type=str, default='kitti', help='kitti/SHIFT')


#Sampler
parser.add_argument('--n_sample', type=int, default=19000, help='Number of sample point')
parser.add_argument('--alpha', type=float, default=0.2, help='Number of sample point')
parser.add_argument('--beta', type=int, default=10, help='Number of sample point')
parser.add_argument('--gama', type=int, default= 0, help='Number of sample point') # TODO delete
parser.add_argument('--sampler_input', type=str, default= 'sparse_input', help='sparse_input/gt/predict_from_past/pseudo_gt')
parser.add_argument('--past_inputs', type=int, default=0, help='Number of past depths inputs which freated depth predictaion')



# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')
parser.add_argument('--max_depth', type=float, default=85.0, help='maximum depth of LIDAR input')
parser.add_argument('--sparse_val', type=float, default=0.0, help='value to endode sparsity with')
parser.add_argument("--rotate", type=str2bool, nargs='?', const=True, default=False, help="rotate image")
parser.add_argument("--flip", type=str, default='hflip', help="flip image: vertical|horizontal")
parser.add_argument("--rescale", type=str2bool, nargs='?', const=True,
                    default=False, help="Rescale values of sparse depth input randomly")
parser.add_argument("--normal", type=str2bool, nargs='?', const=True, default=False, help="normalize depth/rgb input")
parser.add_argument("--no_aug", type=str2bool, nargs='?', const=True, default=False, help="rotate image")
parser.add_argument('--sample_ratio', default=1, type=int, help='Sample ration from the Lidar inputs')


parser.add_argument("--fine_tune", type=str2bool, nargs='?', default=False, help="finetuning sampler and task togheter")
parser.add_argument('--sampler_type', type=str, default='SampleDepth', help='SampleDepth/global_mask')
parser.add_argument("--plot_paper", type=str2bool, nargs='?', const=True,default=False, help="plot image for paper")



# Paths settings
#TODO - remove hard pathes
base_dir_project= '/data/ashomer/project'
parser.add_argument('--save_path', default='{0}/SampleDepth/checkpoints/general_save'.format(base_dir_project), help='save path')
parser.add_argument('--data_path', default='{0}/SampleDepth/Data/'.format(base_dir_project), help='path to desired dataset')
parser.add_argument('--data_path_SHIFT', default='{0}/SHIFT_dataset/discrete/images'.format(base_dir_project), help='path to SHIFT dataset')

#parser.add_argument('--task_weight', default='/home/amitshomer/Documents/SampleDepth/task_checkpoint/SR1/mod_adam_mse_0.001_rgb_batch18_pretrainTrue_wlid0.1_wrgb0.1_wguide0.1_wpred1_patience10_num_samplesNone_multiTrue/model_best_epoch_28.pth.tar', help='path to desired dataset')
# parser.add_argument('--task_weight', default='/home/amitshomer/Documents/SampleDepth/task_checkpoint/SR1_input_gt/mod_adam_mse_0.001_rgb_batch14_pretrainTrue_wlid0.1_wrgb0.1_wguide0.1_wpred1_patience10_num_samplesNone_multiTrue_SR_2/model_best_epoch_28.pth.tar', help='path to desired dataset')
parser.add_argument('--task_weight', default='{0}/SampleDepth/checkpoints/task_checkpoint/kitti_pseudoGT_random_19k/mod_adam_mse_0.002_rgb_batch14_pretrainTrue_wlid0.1_wrgb0.1_wguide0.1_wpred1_patience10_num_samplesNone_multiTrue_SR_2/model_best_epoch_12.pth.tar'.format(base_dir_project), help='path to desired dataset')
parser.add_argument('--eval_path', default='None', help='path to desired pth to eval')
parser.add_argument('--eval_path_random_model', default='None', help='path to desired pth to eval')
parser.add_argument('--eval_path_PredNet', default='None', help='path to desired pth to eval')
parser.add_argument('--eval_path_SampleDepth', default='None', help='path to desired pth to eval')


parser.add_argument('--finetune_path', default='None', help='path to all network for fine tune')
parser.add_argument("--save_pred", type=str2bool, nargs='?', default=False, help="Save the predication as .npz")


# Optimizer settings
parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='plateau', help='{}learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=2, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
parser.add_argument('--gamma', type=float, default=0.5, help='factor to decay learning rate every lr_decay_iters with')

# Loss settings
parser.add_argument('--loss_criterion', type=str, default='mse', choices=allowed_losses(), help="loss criterion")
parser.add_argument('--print_freq', type=int, default=1000, help="print every x iterations")
parser.add_argument('--save_freq', type=int, default=1000, help="save every x interations")
parser.add_argument('--metric', type=str, default='rmse', choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument('--metric_1', type=str, default='mae', choices=allowed_metrics(), help="metric to use during evaluation")
parser.add_argument('--wlid', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wrgb', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wpred', type=float, default=1, help="weight base loss")
parser.add_argument('--wguide', type=float, default=0.1, help="weight base loss")
# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True,
                    default=True, help="cudnn optimization active")
parser.add_argument('--gpu_ids', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument("--gpu_device",type=int, nargs="+", default=[0,1,2,3])

parser.add_argument("--multi", type=str2bool, nargs='?', const=True,
                    default=True, help="use multiple gpus")
parser.add_argument("--seed", type=str2bool, nargs='?', const=True,
                    default=True, help="use seed")
parser.add_argument("--use_disp", type=str2bool, nargs='?', const=True,
                    default=False, help="regress towards disparities")
parser.add_argument('--num_samples', default=0, type=int, help='number of samples')



# distributed training
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--local_rank', dest="local_rank", default=0, type=int)

cuda_send = "cuda:{0}".format(str(parser.parse_args().gpu_device[0]))

def main():
    global args
    args = parser.parse_args()
    if args.num_samples == 0:
        args.num_samples = None
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    # if args.seed:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
                      # 'This will turn on the CUDNN deterministic setting, '
                      # 'which can slow down your training considerably! '
                      # 'You may see unexpected behavior when restarting from checkpoints.')

    # For distributed training
    # init_distributed_mode(args)

    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    ##### define task model trained on Random on eval mode #####
    task_model_random = Models.define_model(mod=args.mod, in_channels=4, thres=args.thres)
    if not args.no_cuda:
        if not args.multi:
            task_model_random = task_model_random.cuda()
        else:
            task_model_random = torch.nn.DataParallel(task_model_random, device_ids = args.gpu_device).to(cuda_send)
    task_model_random.requires_grad_(False)
    task_model_random.eval().to(cuda_send)
    # load weights
    checkpoint = torch.load(args.eval_path_random_model)
    task_model_random.load_state_dict(checkpoint['state_dict'])

    ##### PredNet - prediction net of the next frame #####
    predNet = PredNet(n_sample = args.n_sample, in_channels = 4)

    if not args.no_cuda:
        if not args.multi:
            predNet = predNet.cuda()
        else:
            predNet = torch.nn.DataParallel(predNet, device_ids = args.gpu_device).to(cuda_send)
    predNet.requires_grad_(False)
    predNet.eval().to(cuda_send)
    # load weights
    checkpoint = torch.load(args.eval_path_PredNet)
    predNet.load_state_dict(checkpoint['state_dict'])


    ##### SampleDepth + Task  #####
    task_SampleDepth = Models.define_model(mod=args.mod, in_channels=4, thres=args.thres)
    sampler = SampleDepth(n_sample = args.n_sample, in_channels = 1 if args.sampler_input != 'rgb' else 3)

    if not args.no_cuda:
        if not args.multi:
            task_SampleDepth = task_SampleDepth.cuda()
            sampler = sampler.cuda()
        else:
            task_SampleDepth = torch.nn.DataParallel(task_SampleDepth, device_ids = args.gpu_device).to(cuda_send)
            sampler =torch.nn.DataParallel(sampler,device_ids = args.gpu_device).to(cuda_send)

    task_SampleDepth.requires_grad_(False)
    task_SampleDepth.eval().to(cuda_send)
    sampler.requires_grad_(False)
    sampler.eval().to(cuda_send)

    task_SampleDepth.sampler = sampler

    # load weights
    checkpoint = torch.load(args.eval_path_SampleDepth)
    task_SampleDepth.load_state_dict(checkpoint['state_dict'])

    ## Define the sampler
    # if args.sampler_type == 'SampleDepth':
    #     sampler = SampleDepth(n_sample = args.n_sample, in_channels = 1 if args.sampler_input != 'rgb' else 3)
    #     print("Sampler: SampleDepth")

    # else:
    #     raise Exception("Sampler choosing is not corret")

    # if not args.multi:
    #     sampler = sampler.cuda()
    # else:
    #     sampler =torch.nn.DataParallel(sampler,device_ids = args.gpu_device).to(cuda_send)

    # sampler.requires_grad_(False)
    # sampler.eval()

    # Attach sampler to task_model

    # task_model.sampler = sampler


    #learnable_params = [x for x in task_model.parameters() if x.requires_grad]

    # INIT optimizer/scheduler/loss criterion
    # optimizer = define_optim(args.optimizer, learnable_params, args.learning_rate, args.weight_decay)
    # scheduler = define_scheduler(optimizer, args)

    save_id = '{}_{}_{}_{}_batch{}_pretrain{}_wlid{}_wrgb{}_wguide{}_wpred{}_patience{}_num_samples{}_multi{}_aplpha{}_beta{}_end_to_end'.\
            format(args.mod, args.loss_criterion,
                    args.learning_rate,
                    args.input_type, 
                    args.batch_size,
                    args.pretrained, args.wlid, args.wrgb, args.wguide, args.wpred, 
                    args.lr_decay_iters, args.n_sample, args.multi, args.alpha, args.beta)


    # Optional to use different losses
    criterion_local = define_loss(args.loss_criterion)
    criterion_lidar = define_loss(args.loss_criterion)
    criterion_rgb = define_loss(args.loss_criterion)
    criterion_guide = define_loss(args.loss_criterion)

    # INIT dataset
    if args.dataset =='kitti':
        data_path = args.data_path
    else:
        data_path = args.data_path_SHIFT
    
    dataset = Datasets.define_dataset(args.dataset, data_path, args.input_type, args.side_selection)
    dataset.prepare_dataset(past_inputs=args.past_inputs, plot_paper=args.plot_paper, sampler_input=args.sampler_input)
    #dataset.prepare_dataset()

    train_loader, valid_loader, valid_selection_loader = get_loader(args, dataset, past_inputs=args.past_inputs)


    # Only evaluate
    print("Evaluate only")
    # best_file_lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
    best_file_lst = []
    best_file_lst.append(args.eval_path_random_model)
    if len(best_file_lst) != 0:
        best_file_name = best_file_lst[0]
        print(best_file_name)
        if os.path.isfile(best_file_name):
            sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
            print("=> loading checkpoint '{}'".format(best_file_name))
            checkpoint = torch.load(best_file_name)
            task_model_random.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(best_file_name))
    else:
        print("=> no checkpoint found at due to empy list in folder {}".format(args.save_path))
    
    
    if  args.dataset == 'kitti' :
        validate(valid_selection_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args)
    else: 
        validate(valid_loader, model_random=task_model_random, predNet=predNet, task_SampleDepth=task_SampleDepth, args = args)


def validate(loader, model_random, predNet, task_SampleDepth, args, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    task_loss = AverageMeter()
    samp_loss = AverageMeter() 
    metric = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)
    metric1 = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)

    score = AverageMeter()
    score_1 = AverageMeter()
    # Evaluate model
    model_random.eval()

    # model.sampler.eval()
    list_n_pooints = []
    time_list = []

    pred_next_frame_rmse = []
    dept_reconstruction_rmse = [ ] 
    sceene_num = 1
    past_reconstruct = []
    first_frames= ['00000000_img_front','00000010_img_front','00000020_img_front','00000030_img_front']
    seq_list =[]
    all_seq_list =[]
    # Only forward pass, hence no grads needed
    # ablation 
    rand_hist = torch.zeros(18)
    samp_hist = torch.zeros(18)

    with torch.no_grad():
        # end = time.time()
        for i, (input, gt, predict_input,_) in tqdm(enumerate(loader)):
     
            if not args.no_cuda:
                if isinstance(predict_input, list):
                    input, gt = input.to(cuda_send), gt.to(cuda_send)
                    name = predict_input
                else:
                    input, gt, predict_input = input.to(cuda_send), gt.to(cuda_send), predict_input.to(cuda_send)
            
                start_samp = time.time()
            
            current_frame_name = predict_input[0][predict_input[0].find('/')+1:]
            # Random sampling for 4 first frames
            if current_frame_name in first_frames:
                # new sequence - Reset the memory
                if current_frame_name == '00000000_img_front': 
                    past_reconstruct =[]
                    if len(seq_list)!=0:
                        all_seq_list.append(seq_list)
                        seq_list = []

                input[:,0,:,:] = sample_random(gt.squeeze() , ratio = None, n_sample = args.n_sample, sample_factor_type ='n_points', batch_size = args.batch_size, cuda_send= cuda_send)
                list_n_pooints.append(torch.count_nonzero(input[:,0,:,:]).item()/args.batch_size)
                prediction, lidar_out, precise, guide = model_random(input, epoch)

            # Use PredNet predict next frame and SampleDepth
            else:            
                depth_pred = predNet(torch.cat(past_reconstruct, 1))
                sample_out, bin_pred_map, pred_map = task_SampleDepth.sampler(input=depth_pred, sampler_from=gt)
                sample_input = torch.cat((sample_out, input[:,1:4,:,:]), dim = 1)
                prediction, lidar_out, precise, guide = task_SampleDepth(sample_input, epoch)

                

                if len(seq_list)> 25:
                    samp_hist+=torch.histc(sample_out, bins=18, min=0.01, max=85).detach().cpu()
                rand_samp= sample_random(gt.squeeze() , ratio = None, n_sample = args.n_sample, sample_factor_type ='n_points', batch_size = args.batch_size, cuda_send= cuda_send)
                rand_hist+= torch.histc(rand_samp, bins=18, min=0.01, max=85).detach().cpu()
                
                # gt_full[gt_full>200] = 200
                # gt_hist += torch.histc(gt_full, bins=1000, min=0.01, max=1000).detach().cpu()
            
            # Memory last predications
            if len(past_reconstruct) == 4:
                past_reconstruct.pop()
            past_reconstruct.insert(0, prediction)

            if args.plot_paper: 
                if args.sampler_input =='gt':
                    plot_images(rgb=input[:,1:4,:,:], gt=gt, sample_map=sample_out , sceene_num = sceene_num, pred_depth_com =prediction )
                else: # predicted from the past
                    plot_images(rgb=input[:,1:4,:,:], gt=gt, sample_map=sample_out , sceene_num = sceene_num, pred_next_fame= predict_input,pred_depth_com =prediction )

                    sceene_num +=1


            metric.calculate(prediction[:, 0:1], gt)
            score.update(metric.get_metric(args.metric), metric.num)
            score_1.update(metric.get_metric(args.metric_1), metric.num)

      
            seq_list.append(score.val)

            if (i + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                    i+1, len(loader), score=score))
    
        avg_point_per_image = np.mean(list_n_pooints)
        print("avergae time per image:")
        print(np.mean(time_list))
        if args.evaluate:
            print("===> Average RMSE score on validation set is {:.4f}".format(score.avg))
            print("===> Average MAE score on validation set is {:.4f}".format(score_1.avg))
            print("===> Average point per image {:.4f}".format((avg_point_per_image)))
            print("===> Average sample loss on selection images {:.4f}".format((samp_loss.avg)))

    return score.avg, score_1.avg, losses.avg, avg_point_per_image, samp_loss.avg


def save_checkpoint(state, to_copy, epoch):
    filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, filepath)
    if to_copy:
        if epoch > 0:
            lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
            if len(lst) != 0:
                os.remove(lst[0])
        shutil.copyfile(filepath, os.path.join(args.save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
        print("Best model copied")
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


if __name__ == '__main__':
    main()
