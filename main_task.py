
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
from datetime import datetime
from Loss.loss import define_loss, allowed_losses, MSE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import str2bool, define_optim, define_scheduler, \
                        Logger, AverageMeter, first_run, mkdir_if_missing, \
                        define_init_weights, init_distributed_mode, sample_random

# Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--dataset', type=str, default='kitti', help='dataset to work with: kitti/SHIFT')

parser.add_argument('--nepochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--thres', type=int, default=0, help='epoch for pretraining')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')
parser.add_argument('--mod', type=str, default='mod', choices=Models.allowed_models(), help='Model for use')
parser.add_argument('--batch_size', type=int, default=18, help='batch size')
parser.add_argument('--val_batch_size', default=None, help='batch size selection validation set')
parser.add_argument('--learning_rate', metavar='lr', type=float, default=1e-3, help='learning rate')
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
#Data settng Global
parser.add_argument('--dataset_name', type=str, default='kitti', help='kitti/SHIFT')

# Data augmentation settings Kitti
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
parser.add_argument('--sampler_input', type=str, default= 'sparse_input', help='sparse_input/gt/pseudo_gt')
parser.add_argument('--n_sample', type=int, default=19000, help='Number of sample point')
parser.add_argument('--sample_ratio', default=2, type=int, help='Sample ration from the Lidar inputs')
parser.add_argument('--sample_factor_type', type=str, default='ratio', help='ratio/n_points/None')
parser.add_argument('--sample_method', type=str, default='random', help='random/uniform')

# Paths settings

base_dir_project= os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
parser.add_argument('--save_path', default='{0}/SampleDepth/checkpoints/general_save'.format(base_dir_project), help='save path')
parser.add_argument('--data_path', default='{0}/SampleDepth//Data/'.format(base_dir_project), help='path to desired dataset')
parser.add_argument('--data_path_SHIFT', default='{0}/SHIFT_dataset/discrete/images'.format(base_dir_project), help='path to SHIFT dataset')
parser.add_argument('--erfnet_weight', default='{0}/SampleDepth/checkpoints/task_checkpoint/erfnet_pretrained.pth'.format(base_dir_project), help='path to desired dataset')
parser.add_argument('--eval_path', default='None', help='path to desired pth to eval')
parser.add_argument("--save_pred", type=str2bool, nargs='?', default=False, help="Save the predication as .npz")
parser.add_argument('--save_pred_path', default='None', help='path to desired pth to eval')
parser.add_argument('--reconstructed_folder', default='None', help='path to reconstructed maps')
parser.add_argument('--pseudo_kitti_path', default='None', help='path to pseudo KITTI folder')
parser.add_argument('--prediction_folder', default='None', help='path to prediction folder')

# Optimizer settings
parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='plateau', help='{}learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
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
parser.add_argument("--gpu_device",type=int, nargs="+", default=[0,1,2])
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
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn
    # Init model
    channels_in = 1 if args.input_type == 'depth' else 4
    model = Models.define_model(mod=args.mod, in_channels=channels_in, thres=args.thres)
    define_init_weights(model, args.weight_init)
    # Load on gpu before passing params to optimizer
    if not args.no_cuda:
        if not args.multi:
            model = model.to(cuda_send)
        else:
            model = torch.nn.DataParallel(model, device_ids = args.gpu_device).to(cuda_send)


    save_id = '{}_{}_{}_{}_{}_batch{}_pretrain{}_wlid{}_wrgb{}_wguide{}_wpred{}_patience{}_num_samples{}_multi{}_SR_{}'.\
              format(args.mod, args.optimizer, args.loss_criterion,
                     args.learning_rate,
                     args.input_type, 
                     args.batch_size,
                     args.pretrained, args.wlid, args.wrgb, args.wguide, args.wpred, 
                     args.lr_decay_iters, args.num_samples, args.multi,args.sample_ratio)


    # INIT optimizer/scheduler/loss criterion
    optimizer = define_optim(args.optimizer, model.parameters(), args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)

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
    dataset.prepare_dataset()
    train_loader, valid_loader, valid_selection_loader = get_loader(args, dataset)

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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))

    # Only evaluate
    elif args.evaluate:
        print("Evaluate only")
        # best_file_lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
        best_file_lst = []
        best_file_lst.append(args.eval_path)
        if len(best_file_lst) != 0:
            best_file_name = best_file_lst[0]
            print(best_file_name)
            if os.path.isfile(best_file_name):
                sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
                print("=> loading checkpoint '{}'".format(best_file_name))
                checkpoint = torch.load(best_file_name)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
            else:
                print("=> no checkpoint found at '{}'".format(best_file_name))
        else:
            print("=> no checkpoint found at due to empy list in folder {}".format(args.save_path))
        if  args.dataset == 'kitti' :
            validate(valid_loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide)
            validate(valid_selection_loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide)
        else: 
            validate(valid_loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Dataset : {}".format(args.dataset))
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))
    print("Sample_factor_type: {0}".format(args.sample_factor_type))
    if args.sample_factor_type == 'ratio':
        print("Sample ration of: {0}".format(str(args.sample_ratio)))

    if args.sample_factor_type == 'n_points':
        print("Sample n_points : {0}".format(str(args.n_sample)))
        print("Sampler method: {0}".format(args.sample_method))

    print("Sampler input: {0}".format(args.sampler_input))


    # Load pretrained state for cityscapes in GLOBAL net
    if args.pretrained and not args.resume:
        if not args.load_external_mod:
            if not args.multi:
                target_state = model.depthnet.state_dict()
            else:
                target_state = model.module.depthnet.state_dict()
            check = torch.load(args.erfnet_weight)
            for name, val in check.items():
                # Exclude multi GPU prefix
                mono_name = name[7:] 
                if mono_name not in target_state:
                     continue
                try:
                    target_state[mono_name].copy_(val)
                except RuntimeError:
                    continue
            print('Successfully loaded pretrained model')
        else:
            check = torch.load('external_mod.pth.tar')
            lowest_loss_load = check['loss']
            target_state = model.state_dict()
            for name, val in check['state_dict'].items():
                if name not in target_state:
                    continue
                try:
                    target_state[name].copy_(val)
                except RuntimeError:
                    continue
            print("=> loaded EXTERNAL checkpoint with best rmse {}"
                    .format(lowest_loss_load))

    # Start training
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
        score_train = AverageMeter()
        score_train_1 = AverageMeter()
        metric_train = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)

        # Train model for args.nepochs
        model.train()
        list_n_pooints =[]
        # compute timing
        end = time.time()
        flag_print = 0
        # Load dataset
        for i, (input, gt,_, _) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.to(cuda_send), gt.to(cuda_send)
            
            if args.sampler_input == "sparse_input":
                input = input
            elif args.sampler_input == "gt" or args.sampler_input =='pseudo_gt':
                input[:,0,:,:] = gt.squeeze()
            else:
                raise ValueError('input to Sampler is not valid')
            
            # add gaussian noise
            input[:,0,:,:] = torch.randn(input[:,0,:,:].size()).to(cuda_send)*0.1*input[:,0,:,:]+input[:,0,:,:]

            # Sample augmantion from sparse input (lidar)
            if args.sample_factor_type != 'None':
                if args.sample_method =='random':
                    input[:,0,:,:] = sample_random(input[:,0,:,:] , args.sample_ratio, args.n_sample, args.sample_factor_type, args.batch_size)
                else:# half uniform
                    ratio_per_batch = torch.round((torch.count_nonzero(input[:,0,:,:])/(args.batch_size*args.n_sample))).to(torch.int)
                    mask = torch.zeros_like(input[:,0,:,:])
                    mask[:,::ratio_per_batch,:] = 1
                    new_input = input[:,0,:,:] * mask
                    new_input = sample_random(new_input , args.sample_ratio, args.n_sample, args.sample_factor_type, args.batch_size)
                    input[:,0,:,:] = new_input

            list_n_pooints.append(torch.count_nonzero(input[:,0,:,:]).item()/args.batch_size)

            prediction, lidar_out, precise, guide = model(input, epoch)

            loss = criterion_local(prediction, gt)
            loss_lidar = criterion_lidar(lidar_out, gt)
            loss_rgb = criterion_rgb(precise, gt)
            loss_guide = criterion_guide(guide, gt)
            loss = args.wpred*loss + args.wlid*loss_lidar + args.wrgb*loss_rgb + args.wguide*loss_guide

            losses.update(loss.item(), input.size(0))
            metric_train.calculate(prediction[:, 0:1].detach(), gt.detach())
            score_train.update(metric_train.get_metric(args.metric), metric_train.num)
            score_train_1.update(metric_train.get_metric(args.metric_1), metric_train.num)

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       loss=losses,
                       score=score_train))
        avg_point_per_image = np.mean(list_n_pooints)

        print("===> Average RMSE score on training set is {:.4f}".format(score_train.avg))
        print("===> Average MAE score on training set is {:.4f}".format(score_train_1.avg))
        print("===> Average point per on training images {:.4f}".format((avg_point_per_image)))

        # Evaulate model on validation set
        print("=> Start validation set")
        score_valid, score_valid_1, losses_valid, avg_point_per_image_val = validate(valid_loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, epoch)
        print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
        print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))
        print("===> Average point per on validation images {:.4f}".format((avg_point_per_image_val)))

        # Evaluate model on selected validation set
        if args.subset is None and args.dataset == 'kitti' and not args.sampler_input == 'pseudo_gt':
            print("=> Start selection validation set")
            score_selection, score_selection_1, losses_selection, avg_point_per_image_sel = validate(valid_selection_loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, epoch)
            total_score = score_selection
            print("===> Average RMSE score on selection set is {:.4f}".format(score_selection))
            print("===> Average MAE score on selection set is {:.4f}".format(score_selection_1))
            print("===> Average point per on selection images {:.4f}".format((avg_point_per_image_sel)))

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
            'state_dict': model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)
    if not args.no_tb:
        writer.close()


def validate(loader, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    metric = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)
    score = AverageMeter()
    score_1 = AverageMeter()
    # Evaluate model
    model.eval()
    list_n_pooints = []
    # Only forward pass, hence no grads needed
    with torch.no_grad():
        # end = time.time()
        for i, (input, gt, name,_) in tqdm(enumerate(loader)):
            if not args.no_cuda:
                input, gt = input.to(cuda_send), gt.to(cuda_send)
            
            if args.sampler_input == "sparse_input":
                input = input
            elif args.sampler_input == "gt" or args.sampler_input =='pseudo_gt':
                input[:,0,:,:] = gt.squeeze()
            else:
                raise ValueError('input to Sampler is not valid')

            # Sample augmantion from sparse input (lidar)
            if args.sample_factor_type != 'None':
                if args.sample_method =='random':
                    input[:,0,:,:] = sample_random(input[:,0,:,:] , args.sample_ratio, args.n_sample, args.sample_factor_type, args.batch_size)
                else:# half uniform
                    ratio_per_batch = torch.round((torch.count_nonzero(input[:,0,:,:])/(args.batch_size*args.n_sample))).to(torch.int)
                    mask = torch.zeros_like(input[:,0,:,:])
                    mask[:,::ratio_per_batch,:] = 1
                    new_input = input[:,0,:,:] * mask
                    new_input = sample_random(new_input , args.sample_ratio, args.n_sample, args.sample_factor_type, args.batch_size)
                    input[:,0,:,:] = new_input
            
            # add gaussian noise
            input[:,0,:,:] = torch.randn(input[:,0,:,:].size()).to(cuda_send)*0.1*input[:,0,:,:]+input[:,0,:,:]

            list_n_pooints.append(torch.count_nonzero(input[:,0,:,:]).item()/args.batch_size)
            prediction, lidar_out, precise, guide = model(input, epoch)

            loss = criterion_local(prediction, gt, epoch)
            loss_lidar = criterion_lidar(lidar_out, gt, epoch)
            loss_rgb = criterion_rgb(precise, gt, epoch)
            loss_guide = criterion_guide(guide, gt, epoch)
            loss = args.wpred*loss + args.wlid*loss_lidar + args.wrgb*loss_rgb + args.wguide*loss_guide
            losses.update(loss.item(), input.size(0))

            if args.save_pred: 
                folder = name[0][:name[0].rfind('/')]
                file_name= name[0][name[0].rfind('/'):]
                if not os.path.exists(args.save_pred_path + folder):
                    os.makedirs(args.save_pred_path + folder)
                np.savez_compressed(args.save_pred_path + folder +"/"+ file_name , a=prediction.detach().cpu().numpy().astype(np.float16))

            metric.calculate(prediction[:, 0:1], gt)
            score.update(metric.get_metric(args.metric), metric.num)
            score_1.update(metric.get_metric(args.metric_1), metric.num)

            if (i + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       i+1, len(loader), loss=losses,
                       score=score))
        
        avg_point_per_image = np.mean(list_n_pooints)
        if args.evaluate:
            print("===> Average RMSE score on validation set is {:.4f}".format(score.avg))
            print("===> Average MAE score on validation set is {:.4f}".format(score_1.avg))
            print("===> Average point per image {:.4f}".format((avg_point_per_image)))

    return score.avg, score_1.avg, losses.avg, avg_point_per_image


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
