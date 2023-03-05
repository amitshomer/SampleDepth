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
from datetime import datetime
from Models.SampleDepth import SampleDepth
from Models.SampleDepth_intime import SampleDepth_intime

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
parser.add_argument('--task_weight', default='{0}/SampleDepth/checkpoints/task_checkpoint/kitti_pseudoGT_random_19k/mod_adam_mse_0.002_rgb_batch14_pretrainTrue_wlid0.1_wrgb0.1_wguide0.1_wpred1_patience10_num_samplesNone_multiTrue_SR_2/model_best_epoch_12.pth.tar'.format(base_dir_project), help='path to desired dataset')
parser.add_argument('--eval_path', default='None', help='path to desired pth to eval')
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
parser.add_argument("--gpu_device",type=int, nargs="+", default=[0])

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

    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn
    # Init model
    channels_in = 1 if args.input_type == 'depth' else 4
    
    ## define task model on eval mode
    task_model = Models.define_model(mod=args.mod, in_channels=channels_in, thres=args.thres)
    
    if not args.no_cuda:
        if not args.multi:
            task_model = task_model.cuda()
        else:
            task_model = torch.nn.DataParallel(task_model, device_ids = args.gpu_device).to(cuda_send)

    if not args.fine_tune:
        checkpoint = torch.load(args.task_weight)
        task_model.load_state_dict(checkpoint['state_dict'])
        task_model.requires_grad_(False)
        task_model.eval().to(cuda_send)
    else:
        task_model.requires_grad_(True)
        task_model.train().to(cuda_send)

    # Define the sampler
    if args.sampler_type == 'SampleDepth':
        sampler = SampleDepth(n_sample = args.n_sample, in_channels = 1 if args.sampler_input != 'seg' else 4)

        print("Sampler: SampleDepth")
    elif args.sampler_type == 'global_mask':
        sampler = Global_mask(batch_size = args.batch_size, multi = args.multi, dataset=args.dataset)
        print("Sampler: Global_mask")

    else:
        raise Exception("Sampler choosing is not corret")


    
    if not args.multi:
        sampler = sampler.cuda()
    else:
        sampler =torch.nn.DataParallel(sampler,device_ids = args.gpu_device).to(cuda_send)

    sampler.requires_grad_(True)
    sampler.train()

    # Attach sampler to task_model
    task_model.sampler = sampler

    if args.fine_tune:
        print("## fine tuning task with sampler ###")
        print("Load wieght for task with sampler")
        checkpoint = torch.load(args.finetune_path)
        task_model.load_state_dict(checkpoint['state_dict'])

    learnable_params = [x for x in task_model.parameters() if x.requires_grad]

    # INIT optimizer/scheduler/loss criterion
    optimizer = define_optim(args.optimizer, learnable_params, args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)

    save_id = '{}_{}_{}_{}_{}_batch{}_pretrain{}_wlid{}_wrgb{}_wguide{}_wpred{}_patience{}_num_samples{}_multi{}_aplpha{}_beta{}'.\
            format(args.mod, args.optimizer, args.loss_criterion,
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

    train_loader, valid_loader, valid_selection_loader = get_loader(args, dataset, past_inputs=args.past_inputs)

    # Resume training
    best_epoch = 0
    lowest_loss = np.inf
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    log_file_name = 'log_train_start_0.txt'
    args.resume = first_run(args.save_path)
    if args.resume and args.resume_bool and not args.test_mode and not args.evaluate:
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if os.path.isfile(path):
            log_file_name = 'log_train_start_{}.txt'.format(args.resume)
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['loss']
            best_epoch = checkpoint['best epoch']
            task_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))

    # Only evaluate
    if args.evaluate:
        print("Evaluate only")
        best_file_lst = []
        best_file_lst.append(args.eval_path)
        if len(best_file_lst) != 0:
            best_file_name = best_file_lst[0]
            print(best_file_name)
            if os.path.isfile(best_file_name):
                sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
                print("=> loading checkpoint '{}'".format(best_file_name))
                checkpoint = torch.load(best_file_name)
                task_model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(best_file_name))
        else:
            print("=> no checkpoint found at due to empy list in folder {}".format(args.save_path))
       
        if  args.dataset == 'kitti' :
            validate(valid_selection_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args)
            # validate(valid_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args)

        else: 
            validate(valid_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        log_file_name = 'log_train_start_0.txt'
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Dataset : {}".format(args.dataset))
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in task  model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in task_model.parameters())/1e6))
    print("Number of parameters in Sampler  model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in sampler.parameters())/1e6))
    print("Sample ration of: {0}".format(str(args.sample_ratio)))
    print("Alpha factor: {0}".format(str(args.alpha)))
    print("Beta factor: {0}".format(str(args.beta)))
    print("Sampler input: {0}".format(args.sampler_input))

    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start EPOCH {}".format(epoch + 1))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(args.save_path)
        # Adjust learning rate
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))

        # Define container objects
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        task_loss = AverageMeter()
        samp_loss = AverageMeter()
        choice_loss = AverageMeter()


        score_train = AverageMeter()
        score_train_1 = AverageMeter()
        metric_train = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)

        # Train model for args.nepochs
        if args.fine_tune:
            task_model.train() 
        else:
            task_model.eval()
        task_model.sampler.train()


        list_n_pooints =[]
        # compute timing
        end = time.time()
        flag_print = 0
        # Load dataset
        for i, (input, gt, predict_input, seg) in tqdm(enumerate(train_loader)):
            # Time dataloader
            data_time.update(time.time() - end)
            if isinstance(predict_input, list):
                name = predict_input
            else:
                predict_input =  predict_input.to(cuda_send)
            
            if not isinstance(seg, list): 
                seg = seg.to(cuda_send)

            input, gt =  input.to(cuda_send), gt.to(cuda_send)
            
            # if isinstance(predict_input, list) and isinstance(seg, list) :
            #     input, gt = input.to(cuda_send), gt.to(cuda_send)
            #     name = predict_input
            # elif isinstance(predict_input, list):
            #     input, gt, seg = input.to(cuda_send), gt.to(cuda_send), seg.to(cuda_send)
            #     name = predict_input
            # else:
            #     input, gt, predict_input, seg = input.to(cuda_send), gt.to(cuda_send), predict_input.to(cuda_send), seg.to(cuda_send)

            if args.sampler_input == "sparse_input":
                sample_out, bin_pred_map, pred_map = task_model.sampler(input = input[:,0,:,:].unsqueeze(dim =1), sampler_from=input[:,0,:,:].unsqueeze(dim =1)) 
            
            elif args.sampler_input == "gt" or args.sampler_input =='pseudo_gt':
                if args.sampler_type =='SampleDepth':
                    sample_out, bin_pred_map, pred_map = task_model.sampler(input=gt, sampler_from=gt)
                elif args.sampler_type == 'global_mask':
                    sample_out, bin_pred_map, pred_map = task_model.sampler(sampler_from=gt)

            elif args.sampler_input == 'predict_from_past':
                sample_out, bin_pred_map, pred_map = task_model.sampler(input=predict_input, sampler_from=gt)
            
            elif args.sampler_input == 'rgb':
                sample_out, bin_pred_map, pred_map = task_model.sampler(input=input[:,1:4,:,:].detach(), sampler_from=gt)
            
            elif args.sampler_input == 'seg':
                sample_out, bin_pred_map, pred_map = task_model.sampler(input=torch.cat((seg, input[:,1:4,:,:].detach()), dim = 1), sampler_from=gt)


            else:
                raise ValueError('input to Sampler is not valid')

            
            sample_input = torch.cat((sample_out, input[:,1:4,:,:]), dim = 1)
            
            if args.sampler_type == 'global_mask':
                list_n_pooints.append(torch.sum(sample_out[:,0,:,:]>0.001).item()/args.batch_size)
            else:
                list_n_pooints.append(torch.count_nonzero(sample_out[:,0,:,:]).item()/args.batch_size)

            prediction, lidar_out, precise, guide = task_model(sample_input, epoch)


            ## task loss
            loss = criterion_local(prediction, gt)
            loss_lidar = criterion_lidar(lidar_out, gt)
            loss_rgb = criterion_rgb(precise, gt)
            loss_guide = criterion_guide(guide, gt)
            loss_task = args.wpred*loss + args.wlid*loss_lidar + args.wrgb*loss_rgb + args.wguide*loss_guide
            
            # Sampler loss
 
            loss_number_sampler = torch.abs((pred_map.sum()/args.batch_size)-args.n_sample)/args.n_sample

            loss_softarg =torch.zeros(1).to(cuda_send)
            # loss_softarg = task_model.sampler.module.get_softargmax_loss()

            # total loss
            # total_loss = args.alpha * loss_task + args.beta * loss_number_sampler + 0 * loss_softarg
            if args.sampler_type == 'global_mask' and args.dataset =='kitti' and  args.sampler_input !="pseudo_gt" :
                loss_choice = torch.sum((pred_map>0.001)&(gt==0))/args.n_sample                
                loss_choice_scalar = loss_choice.item()
                choice_loss.update(loss_choice_scalar, input.size(0))
                total_loss = 1* loss_task +  10* loss_number_sampler + 0 * loss_softarg +10* loss_choice #TODO - change hard coded


            else: # SampleDepth or GlobalMask in SHIFT dataset
                loss_choice_scalar = None
                total_loss = args.alpha * loss_task + args.beta * loss_number_sampler + 0 * loss_softarg

            
            #     loss_global_mask = task_model.sampler.module.global_mask_loss()
            #     total_loss = total_loss + loss_global_mask/200                
            #     # total_loss = total_loss 

            #     if i % 200 ==0 :
            #         print("Loss global mask: {0} ".format(str(loss_global_mask.item())))

            if i % 500 ==0 :
                print("Loss task: {0} , Loss number sample: {1}, Loss choice map: {2}, Loss softargmax: {3}, Total loss: {4}".format(str(loss_task.item()),
                                                                                                        str(loss_number_sampler.item()),
                                                                                                        str(loss_choice_scalar),
                                                                                                        str(loss_softarg.item()),
                                                                                                        str(total_loss.item()) ))
                print("Number of sample lat iter: {0}".format(list_n_pooints[-1]))
            
            losses.update(total_loss.item(), input.size(0))
            task_loss.update(loss_task.item(), input.size(0)) # TODO - cgeck size0 
            samp_loss.update(loss_number_sampler.item(), input.size(0))


            metric_train.calculate(prediction[:, 0:1].detach(), gt.detach())
            score_train.update(metric_train.get_metric(args.metric), metric_train.num)
            score_train_1.update(metric_train.get_metric(args.metric_1), metric_train.num)

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            # if args.clip_grad_norm != 0:
            #    nn.utils.clip_grad_norm(task_model.parameters(), args.clip_grad_norm)
                #nn.utils.clip_grad_norm(task_model.sampler.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Toatal Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Task Loss {task_loss.val:.4f} ({task_loss.avg:.4f})\t'
                    'Samp Loss {samp_loss.val:.4f} ({samp_loss.avg:.4f})\t'
                    'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                    epoch+1, i+1, len(train_loader), batch_time=batch_time,
                    loss=losses,
                    task_loss = task_loss, 
                    samp_loss = samp_loss,
                    score=score_train))
                if args.sampler_type == 'global_mask':
                    print('Choice Loss {choice_loss.val:.4f} ({choice_loss.avg:.4f})'.format(choice_loss = choice_loss))

            # # print('nnnn')
            if i == 4000:
                break
        avg_point_per_image = np.mean(list_n_pooints)

        print("===> Average RMSE score on training set is {:.4f}".format(score_train.avg))
        print("===> Average MAE score on training set is {:.4f}".format(score_train_1.avg))
        print("===> Average point per on training images {:.4f}".format(avg_point_per_image))

        #Evaulate model on validation set
        # print("=> Start validation set")
        # score_valid, score_valid_1, losses_valid, avg_point_per_image_val, _ = validate(valid_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args, epoch)
        # print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
        # print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))
        # print("===> Average point per on validation images {:.4f}".format((avg_point_per_image_val)))

        # Evaluate model on selected validation set
        if args.subset is None and args.dataset == 'kitti' and ((not args.sampler_input == 'pseudo_gt') and (not args.sampler_input == 'predict_from_past')):
            print("=> Start selection validation set")
            score_selection, score_selection_1, losses_selection, avg_point_per_image_sel, avg_sample_loss_val = validate(valid_selection_loader, task_model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args, epoch)
            total_score = score_selection
            print("===> Average RMSE score on selection set is {:.4f}".format(score_selection))
            print("===> Average MAE score on selection set is {:.4f}".format(score_selection_1))
            print("===> Average point per on selection images {:.4f}".format((avg_point_per_image_sel)))
            print("===> Average sample loss on selection images {:.4f}".format((avg_sample_loss_val)))


        else:
            total_score = score_valid

        print("===> Last best score was RMSE of {:.4f} in epoch {}".format(lowest_loss,
                                                                           best_epoch))
        # Adjust lr if loss plateaued
        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))

        # File to keep latest epoch
        with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
            f.write(str(epoch))

        # Save model
        to_save = False
        if total_score < lowest_loss:

            to_save = True
            best_epoch = epoch+1
            lowest_loss = total_score
        save_checkpoint({
            'epoch': epoch + 1,
            'best epoch': best_epoch,
            'arch': args.mod,
            'state_dict': task_model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)

        if args.sampler_type == 'global_mask':
            print("Save .npz pred map")
            numpy_to_save = pred_map.detach().cpu().numpy()
            save_pred_path = os.path.join(args.save_path, 'pred_map_epoch_{0}.npz'.format(epoch))
            np.savez(save_pred_path, name1 = numpy_to_save)

    if not args.no_tb:
        writer.close()


def validate(loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, args, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    task_loss = AverageMeter()
    samp_loss = AverageMeter() 
    metric = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)
    metric1 = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)

    score = AverageMeter()
    score_1 = AverageMeter()
    # Evaluate model
    model.eval()
    model.sampler.eval()
    list_n_pooints = []
    time_list = []

    pred_next_frame_rmse = []
    dept_reconstruction_rmse = [ ] 
    sceene_num = 1

    with torch.no_grad():
        for i, (input, gt, predict_input, seg) in tqdm(enumerate(loader)):
            # # to do- delete
            # base_path = '/data/ashomer/project/SHIFT_dataset/pred_sample/val/'
            # folder = name[0][:name[0].rfind('/')]
            # file_name= name[0][name[0].rfind('/'):]

            # if not os.path.exists(base_path + folder +"/"+ file_name+".npz"):
            if not args.no_cuda:
                input, gt = input.to(cuda_send), gt.to(cuda_send)

                if isinstance(predict_input, list):
                    name = predict_input
                else:
                    predict_input = predict_input.to(cuda_send)
                if not isinstance(seg, list):
                    seg = seg.to(cuda_send)
            
                start_samp = time.time()

                
            if args.sampler_input == "sparse_input":
                sample_out, bin_pred_map, pred_map = model.sampler(input = input[:,0,:,:].unsqueeze(dim =1), sampler_from=input[:,0,:,:].unsqueeze(dim =1)) 
           
            elif args.sampler_input == "gt" or args.sampler_input =='pseudo_gt':
                if args.sampler_type =='SampleDepth':
                    sample_out, bin_pred_map, pred_map = model.sampler(input=gt, sampler_from=gt)
                elif args.sampler_type == 'global_mask':
                    sample_out, bin_pred_map, pred_map = model.sampler(sampler_from=gt)

            elif args.sampler_input == 'predict_from_past':
                sample_out, bin_pred_map, pred_map = model.sampler(input=predict_input, sampler_from=gt)
            
            elif args.sampler_input == 'rgb':
                sample_out, bin_pred_map, pred_map = model.sampler(input=input[:,1:4,:,:].detach(), sampler_from=gt)

            elif args.sampler_input == 'seg':
                # sample_out, bin_pred_map, pred_map = model.sampler(input=seg, sampler_from=gt)
                sample_out, bin_pred_map, pred_map = model.sampler(input=torch.cat((seg, input[:,1:4,:,:].detach()), dim = 1), sampler_from=gt)

            else:
                raise ValueError('input to Sampler is not valid')

            # # # # ### try
            # sample_out[sample_out[:,:,:,:]<0.1]=0            
            # sample_out = sample_random(sample_out , None, 4300, 'n_points' , 1)

            ### try
            # _,_,H,W = sample_out.shape
            # exect_number = 4096
            # if torch.count_nonzero(sample_out).item() > exect_number :
            #     new = torch.zeros(sample_out.shape).cuda()
            #     new[sample_out!= 0]= 1
            #     grade_pixel = bin_pred_map[:,1,:,:]*new.cuda()
            #     flat_grade_piexel = grade_pixel.view(-1)
            #     hight_indexes = torch.topk(flat_grade_piexel, exect_number)[1]
            #     sample_out_new=torch.zeros(sample_out.view(-1).shape)
            #     sample_out_new[hight_indexes]= 1
            #     sample_out_new = sample_out_new.view(H,W)
            #     sample_out = sample_out* sample_out_new.cuda()
            
            # else: 
            #     delta= exect_number-torch.count_nonzero(sample_out).item()
            #     sample_out_new2=torch.zeros(sample_out.view(-1).shape)
            #     new2 = torch.zeros(sample_out.shape).cuda()
            #     new2[sample_out == 0]= 1
            #     grade_pixel2 = bin_pred_map[:,0,:,:]*new2.cuda()
            #     flat_grade_piexel2 = grade_pixel2.view(-1)
            #     hight_indexes2 = torch.topk(flat_grade_piexel2, exect_number, largest= False)[1]
            #     sample_out_new2[hight_indexes2]= 1
            #     sample_out_new2 = sample_out_new2.view(H,W) 
            #     mask= (sample_out.squeeze()==0)&(sample_out_new2.cuda()>0)

            #     sample_out = sample_out +  gt*sample_out_new2.cuda()*mask
            # print(torch.count_nonzero(sample_out))
            # #####
                
                
                
            end_samp = time.time()
            time_list.append(end_samp - start_samp)

            sample_input = torch.cat((sample_out, input[:,1:4,:,:]), dim = 1)
                    
            if args.sampler_type == 'global_mask':
                list_n_pooints.append(torch.sum(sample_out[:,0,:,:]>0.0001).item()/args.batch_size)
            else:
                list_n_pooints.append(torch.count_nonzero(sample_out[:,0,:,:]).item()/args.batch_size)

            prediction, lidar_out, precise, guide = model(sample_input, epoch)

            # save locally npz - TODO delete after 
            # if not os.path.exists(base_path+folder):
            #     os.makedirs(base_path+folder)
            #     # mask= sample_out[0,0,:,:] > 0.001
            #     # indices = mask.nonzero()
            # np.savez_compressed(base_path + folder +"/"+ file_name , a=pred_map.type(torch.int).detach().cpu().numpy())  

            loss = criterion_local(prediction, gt, epoch)
            loss_lidar = criterion_lidar(lidar_out, gt, epoch)
            loss_rgb = criterion_rgb(precise, gt, epoch)
            loss_guide = criterion_guide(guide, gt, epoch)
            loss_task = args.wpred*loss + args.wlid*loss_lidar + args.wrgb*loss_rgb + args.wguide*loss_guide
            
            # Sampler loss
            # loss_number_sampler = model.sampler.module.sample_number_loss(bin_pred_map)
            loss_number_sampler = torch.abs((pred_map.sum()/args.batch_size)-args.n_sample)/args.n_sample

            loss_softarg =torch.zeros(1).to(cuda_send)
            # loss_softarg = model.sampler.module.get_softargmax_loss()
            
    
            # total loss
            total_loss = 0.2 * loss_task + 1 * loss_number_sampler + 0 * loss_softarg
            
            # if args.sampler_type == 'global_mask':
            #     loss_global_mask = model.sampler.module.global_mask_loss()
            #     total_loss = total_loss + 0.2* loss_global_mask
            # if i % 100 ==0 :
            #     print("Loss task: {0} , Loss number sample:{1}, Loss softargmax {2}, Total loss: {3}".format(str(loss_task.item()),
            #                                                                                             str(loss_number_sampler.item()),
            #                                                                                             str(loss_softarg.item()),
            #                                                                                             str(total_loss.item()) ))
            if args.save_pred: 
                base_path = '/data/ashomer/project/SampleDepth/Data/pred_sample/'
                folder = name[0][:name[0].rfind('/')]
                file_name= name[0][name[0].rfind('/'):]
                if not os.path.exists(base_path+folder):
                    os.makedirs(base_path+folder)
                np.savez_compressed(base_path + folder +"/"+ file_name , a=prediction.detach().cpu().numpy().astype(np.float16))
            if args.plot_paper: 
                if args.sampler_input =='gt':
                    plot_images(rgb=input[:,1:4,:,:], gt=gt, sample_map=sample_out , sceene_num = sceene_num, pred_depth_com =prediction,lidar = input[:,0,:,:] )
                else: # predicted from the past
                    plot_images(rgb=input[:,1:4,:,:], gt=gt, sample_map=sample_out , sceene_num = sceene_num, pred_next_fame= predict_input,pred_depth_com =prediction )

                sceene_num +=1

            losses.update(total_loss.item(), input.size(0))
            task_loss.update(loss_task.item(), input.size(0)) # TODO - cgeck size0 
            samp_loss.update(loss_number_sampler.item(), input.size(0))

            metric.calculate(prediction[:, 0:1], gt)
            score.update(metric.get_metric(args.metric), metric.num)
            score_1.update(metric.get_metric(args.metric_1), metric.num)

            ## added
            # pred_next_frame_rmse = []
            # metric1.calculate(predict_input, gt)
            # dept_reconstruction_rmse.append(metric.get_metric(args.metric)) 
            # pred_next_frame_rmse.append(metric1.get_metric(args.metric))


            if (i + 1) % args.print_freq == 0:
    
                print('Test: [{0}/{1}]\t'
                    'Toatal {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Task Loss {task_loss.val:.4f} ({task_loss.avg:.4f})\t'
                    'Samp Loss {samp_loss.val:.4f} ({samp_loss.avg:.4f})\t'
                    'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                    i+1, len(loader), loss=losses,task_loss = task_loss, samp_loss = samp_loss,
                    score=score))
    
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
