"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (35, 30)
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import torch.optim
from torch.optim import lr_scheduler
import errno
import sys
from torchvision import transforms
import torch.nn.init as init
import torch.distributed as dist
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as npl
from PIL import Image


def plot_images(rgb, gt, sample_map , sceene_num, pred_next_fame = None, pred_depth_com = None):
    sceene = "to_pap_{0}".format(sceene_num)
    general_path= "/data/ashomer/project/SampleDepth/visual/{0}/".format(sceene )
    if not os.path.exists(general_path):
        os.makedirs(general_path)
    
    ##### save gt map
    if pred_next_fame is not None:
        pred_nepred_depth_com_i =pred_depth_com.squeeze().detach().cpu().numpy() 
        pred_nepred_depth_com_i = convert_depth_to_rgb(pred_nepred_depth_com_i)
        im = Image.fromarray(pred_nepred_depth_com_i)
        path = general_path +'depth_comp_pred_last_stage.png'
        im = im.convert("RGB")
        im.save(path)
        
        pred_next_fame_i=pred_next_fame.squeeze().detach().cpu().numpy() 
        pred_next_fame_i = convert_depth_to_rgb(pred_next_fame_i)
        im = Image.fromarray(pred_next_fame_i)
        path = general_path +'pred_next.png'
        im = im.convert("RGB")
        im.save(path)
        
        gt_i=gt.squeeze().detach().cpu().numpy() 
        gt_i = convert_depth_to_rgb(gt_i)
        im = Image.fromarray(gt_i)
        path = general_path +'gt.png'
        im = im.convert("RGB")
        im.save(path)
        
        ##### sample_out gt 
        sample_map_i= gt_i.copy()
        sample_map_i[sample_map.squeeze().detach().cpu().numpy()==0] = 0
        im = Image.fromarray(sample_map_i)
        path = general_path +'sample_pred_next.png'
        im = im.convert("RGB")
        im.save(path) 
        
    else:    

                
        pred_nepred_depth_com_i =pred_depth_com.squeeze().detach().cpu().numpy() 
        pred_nepred_depth_com_i = convert_depth_to_rgb(pred_nepred_depth_com_i)
        im = Image.fromarray(pred_nepred_depth_com_i)
        path = general_path +'depth_comp_pred.png'
        im = im.convert("RGB")
        im.save(path)

        gt_i=gt.squeeze().detach().cpu().numpy() 
        gt_i = convert_depth_to_rgb(gt_i)
        im = Image.fromarray(gt_i)
        path = general_path +'gt.png'
        im = im.convert("RGB")
        im.save(path)

        ##### RGB
        rgb_im=rgb.squeeze().permute(1,2,0).detach().cpu().numpy() 
        im = Image.fromarray(rgb_im.astype('uint8')).convert('RGB')
        path =general_path + 'rgb.png'
        im.save(path)

        ##### sample_out gt 
        sample_map_i= gt_i.copy()
        sample_map_i[sample_map.squeeze().detach().cpu().numpy()==0] = 0
        im = Image.fromarray(sample_map_i)
        path = general_path +'sample_gt.png'
        im = im.convert("RGB")
        im.save(path)


def convert_depth_to_rgb(input):
    # vmax = np.percentile(input,85)      
    norm= npl.colors.Normalize(vmin = 0,vmax= 85)
    mapper = cm.ScalarMappable(norm=norm, cmap="Spectral")
    color_im = (mapper.to_rgba(input)[:,:,:3]*255).astype(np.uint8)
    return color_im


def sample_random(depth, ratio, n_sample, sample_factor_type, batch_size):
    torch.seed()

    mask_keep = depth > 0
    depth_sampled = torch.zeros(depth.shape).cuda()
    # prob = float(self.num_samples) / n_keep
    
    if sample_factor_type =='ratio':
        prob = 1/ratio
    elif sample_factor_type == 'n_points':
        prob = (batch_size*n_sample)/torch.count_nonzero(depth)

    mask_keep =  torch.bitwise_and(mask_keep, torch.rand(depth.shape).cuda() < prob)
    depth_sampled[mask_keep] = depth[mask_keep]
    return depth_sampled


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'AdamW':
        optimizer=  torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def first_run(save_path):
    txt_file = os.path.join(save_path, 'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return ''
        return saved_epoch
    return ''


def depth_read(img, sparse_val, dataset ='kitti',max_depth = 80):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    if dataset =='kitti':
        depth_png = np.array(img, dtype=int)
        depth_png = np.expand_dims(depth_png, axis=2)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = sparse_val
    else:
        depth_png = np.array(img, dtype=np.float32)    
        DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float32)
        depth = (256 * 256 * depth_png[:, :, 2] +  256 * depth_png[:, :, 1] + depth_png[:, :, 0]) * DEPTH_C  # in meters  
        depth[depth>max_depth] = max_depth
        im = Image.fromarray(depth)
        im = im.resize((640, 400), Image.NEAREST) # TODO- change hard coded
        depth = np.array(im, dtype=np.float32)    

    return depth


class show_figs():
    def __init__(self, input_type, savefig=False):
        self.input_type = input_type
        self.savefig = savefig

    def save(self, img, name):
        img.save(name)

    def transform(self, input, name='test.png'):
        if isinstance(input, torch.tensor):
            input = torch.clamp(input, min=0, max=255).int().cpu().numpy()
            input = input * 256.
            img = Image.fromarray(input)

        elif isinstance(input, np.array):
            img = Image.fromarray(input)

        else:
            raise NotImplementedError('Input type not recognized type')

        if self.savefig:
            self.save(img, name)
        else:
            return img

# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def write_file(content, location):
    file = open(location, 'w')
    file.write(str(content))
    file.close()


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def save_fig(inp, name='saved.png'):
    if isinstance(inp, torch.Tensor):
        # inp = inp.permute([2, 0, 1])
        inp = transforms.ToPILImage()(inp.int())
        inp.save(name)
        return
    pil = Image.fromarray(inp)
    pil.save(name)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # Does not seem to work?
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
