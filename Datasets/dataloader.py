"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import depth_read


def get_loader(args, dataset, past_inputs = 0, past_input_path =''):
    """
    Define the different dataloaders for training and validation
    """

    if args.dataset =='kitti':
        data_path = args.data_path
        crop_size = (args.crop_h, args.crop_w)
        perform_transformation = not args.no_aug
    else:
        data_path = args.data_path_SHIFT
        crop_size = (800, 1280)
        perform_transformation = False

    train_dataset = Dataset_loader(
            data_path, dataset.train_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=perform_transformation, num_samples=args.num_samples, dataset=args.dataset,
            past_inputs=past_inputs, past_input_path=past_input_path, sampler_input=args.sampler_input )
    val_dataset = Dataset_loader(
            data_path, dataset.val_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples, dataset=args.dataset, past_inputs=past_inputs,
            past_input_path=past_input_path, sampler_input=args.sampler_input)
    if args.dataset =='kitti':
        val_select_dataset = Dataset_loader(
                data_path, dataset.selected_paths, args.input_type,
                resize=None, rotate=args.rotate, crop=crop_size,
                flip=args.flip, rescale=args.rescale, max_depth=args.max_depth,
                sparse_val=args.sparse_val, normal=args.normal, 
                disp=args.use_disp, train=False, num_samples=args.num_samples, dataset=args.dataset )

    train_sampler = None
    val_sampler = None
    if args.subset is not None:
        random.seed(1)
        train_idx = [i for i in random.sample(range(len(train_dataset)-1), args.subset)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        random.seed(1)
        val_idx = [i for i in random.sample(range(len(val_dataset)-1), round(args.subset*0.5))]
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=int(args.val_batch_size),  sampler=val_sampler,
        shuffle=False, num_workers=args.nworkers_val,
        pin_memory=True, drop_last=True)
    if args.dataset == 'kitti':
        val_selection_loader = DataLoader(
            val_select_dataset, batch_size=int(args.val_batch_size), shuffle=False,
            num_workers=args.nworkers_val, pin_memory=True, drop_last=True)
    else:
        val_selection_loader = None
    return train_loader, val_loader, val_selection_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, data_path, dataset_type, input_type, resize,
                 rotate, crop, flip, rescale, max_depth, dataset, sparse_val=0.0, 
                 normal=False, disp=False, train=False, num_samples=None, past_inputs = 0, past_input_path ='',sampler_input = 'gt'):

        # Constants
        self.use_rgb = input_type == 'rgb'
        self.datapath = data_path
        self.dataset_type = dataset_type
        self.train = train
        self.resize = resize
        self.flip = flip
        self.crop = crop
        self.rotate = rotate
        self.rescale = rescale
        self.max_depth = max_depth
        self.sparse_val = sparse_val
        self.dataset =dataset
        self.past_inputs = past_inputs
        self.past_input_path = past_input_path
        self.sampler_input = sampler_input

        # Transformations
        self.totensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(size=crop)

        # Names
        self.img_name = 'img'
        self.lidar_name = 'lidar_in' 
        self.gt_name = 'gt' 
        self.seg_name = 'seg'

        # Define random sampler
        self.num_samples = num_samples


    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['gt'])


    def define_transforms(self, input, gt, img=None, seg=None):
        # Define random variabels
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'

        if self.train: #right now transformation not supporeted in SHIFT dataset
            i, j, h, w = transforms.RandomCrop.get_params(input, output_size=self.crop)
            input = F.crop(input, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)
            if hflip_input:
                input, gt = F.hflip(input), F.hflip(gt)

            if self.use_rgb:
                img = F.crop(img, i, j, h, w)
                if hflip_input:
                    img = F.hflip(img)
            input, gt = depth_read(input, self.sparse_val), depth_read(gt, self.sparse_val)
            
        else:
            if self.dataset =='kitti':
                input, gt = self.center_crop(input), self.center_crop(gt)
            
            if self.use_rgb and self.dataset =='kitti':
                img = self.center_crop(img)
            
            if self.dataset =='kitti':
                input, gt = depth_read(input, self.sparse_val, dataset = 'kitti'), depth_read(gt, self.sparse_val, dataset = 'kitti')
            else: 
                # depth read
                gt = depth_read(gt, self.sparse_val, dataset = 'SHIFT',max_depth =self.max_depth)
                # make input like LiDAR unfirom pattern
                input = np.zeros_like(gt)
                input[::7,::2] = 1
                input = input * gt  

                #seg read - right now transformation not supporeted in SHIFT dataset
                if seg != None:
                    seg = np.array(seg, dtype=int)

        return input, gt, img, seg

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        if  self.dataset == 'kitti':
            sparse_depth_name = os.path.join(self.dataset_type[self.lidar_name][idx])
            with open(sparse_depth_name, 'rb') as f:
                sparse_depth = Image.open(f)
                w, h = sparse_depth.size
                sparse_depth = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w)
        else:
            sparse_depth = None
        
        gt_name = os.path.join(self.dataset_type[self.gt_name][idx])
        with open(gt_name, 'rb') as f:
            if  self.dataset == 'kitti':
                gt = Image.open(f)
                w, h = gt.size
                gt = F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
            else:
                gt = (Image.open(f).convert('RGB'))
                w, h = gt.size
                gt = F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
                


        img = None
        # RGB load
        if self.use_rgb:
            img_name = self.dataset_type[self.img_name][idx]
            with open(img_name, 'rb') as f:
                img = (Image.open(f).convert('RGB'))

            img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)
            if self.dataset == 'SHIFT':
                img = img.resize((640, 400),Image.LANCZOS) 
      
        # segmantion load 
        if self.dataset == 'SHIFT':
            seg_name = self.dataset_type[self.seg_name][idx]
            with open(seg_name, 'rb') as f:
                seg = (Image.open(f).convert('RGB'))
            seg = seg.resize((640, 400),Image.NEAREST) 
        else:
            seg = None # not supported yet
        # trasmormation
        sparse_depth_np, gt_np, img_pil, seg = self.define_transforms(sparse_depth, gt, img, seg)
        input, gt = self.totensor(sparse_depth_np).float(), self.totensor(gt_np).float()
        
        if seg is not None:
            seg = self.totensor(seg[:,:,0]).float()

        if self.use_rgb:
            img_tensor = self.totensor(img_pil).float()
            img_tensor = img_tensor*255.0
            input = torch.cat((input, img_tensor), dim=0)
        
        if (self.past_inputs != 0 and self.sampler_input != 'predict_from_past') or self.sampler_input=='pseudo_gt':
            ## right now only support t-1 past input
            if self.dataset =='SHIFT':
                full_file_path = self.dataset_type[self.gt_name][idx]
                val_or_train = full_file_path[full_file_path.find('images')+7: full_file_path.find('front')- 1]
                file_name = full_file_path[full_file_path.rfind('/'):]
                
                #Current predicated depth loader
                file_index = str(int(file_name[1: file_name.find('_')])).rjust(8,'0')
                sceene_folder = full_file_path[full_file_path.find('front')+ 6: full_file_path.rfind('/')]
                indices_base_path = '/data/ashomer/project/SHIFT_dataset/pred_sample/'
                past_data_path = indices_base_path + val_or_train + '/' + sceene_folder + '/'+ file_index+'_img_front.npz'
                
                # if os.path.exists(past_data_path):
                #     with np.load(past_data_path, allow_pickle=True) as data:
                #         indicies_current_predmap = data['a'].squeeze()
                #         indicies_current_predmap = self.totensor(indicies_current_predmap).float()
                        # size_tens= indicies_current_predmap.shape[1] - 1
                        # pad = torch.zeros(1,100000-size_tens,2)
                        # last_element =  torch.tensor([[[size_tens, size_tens]]])
                        # big_indices = torch.cat((indicies_current_predmap, pad, last_element), dim=1)

                # Past depth loader
                for i in range(1, self.past_inputs+1):
                    file_index = str(int(file_name[1: file_name.find('_')]) - 10 *i).rjust(8,'0')
                    sceene_folder = full_file_path[full_file_path.find('front')+6: full_file_path.rfind('/')]
                    past_data_path = self.past_input_path + val_or_train + '/' + sceene_folder + '/'+ file_index+'_img_front.npz'
                    if os.path.exists(past_data_path):
                        with np.load(past_data_path, allow_pickle=True) as data:
                            past_depth = data['a'].squeeze()
                            past_depth = self.totensor(past_depth).float()
                            if i == 1:
                                past_depths= past_depth
                            else:
                                past_depths = torch.cat((past_depths,past_depth),0)
                    else: 
                        raise Exception("No past data .npz file in {0}".format(past_data_path))
            
            else: # kitti dataset
                full_file_path = self.dataset_type[self.img_name][idx]
                file_name = full_file_path[full_file_path.rfind('/'):]
                file_index = str(int(file_name[1: file_name.find('.png')])).rjust(10,'0')
                base_pass_path = '/data/ashomer/project/SampleDepth/Data/pred_sample/'
                if not self.sampler_input=='pseudo_gt':
                    # past_data_path = base_pass_path + full_file_path[full_file_path.find('Data')+5:full_file_path.rfind(".png")]+".npz"
                    for i in range(1, self.past_inputs+1):
                        file_index = str(int(file_name[1: file_name.find('.png')])-i).rjust(10,'0')
                        past_data_path = base_pass_path + full_file_path[full_file_path.find('Data')+5:full_file_path.rfind("data/")+5]+ file_index +".npz"
                        if os.path.exists(past_data_path):
                            # print(past_data_path)

                            with np.load(past_data_path, allow_pickle=True) as data:
                                past_depth = data['a'].squeeze()
                                past_depth = self.totensor(past_depth).float()
                                if i == 1:
                                    past_depths= past_depth
                                else:
                                    past_depths = torch.cat((past_depths,past_depth),0)
                        else: 
                            raise Exception("No past data .npz file in {0}".format(past_data_path))

                ### the gt is pseudo gt
                file_name_pseudo_gt = full_file_path[full_file_path.rfind('/'):]
                pseudo_gt_base_path = '/data/ashomer/project/SampleDepth/Data/pseudo_gt/'
                past_pseudogt_data_path = pseudo_gt_base_path + full_file_path[full_file_path.find('Data')+5:full_file_path.rfind(".png")]+".npz"
                if os.path.exists(past_pseudogt_data_path):
                        # print(past_pseudogt_data_path)
                        with np.load(past_pseudogt_data_path, allow_pickle=True) as data:
                            gt = data['a'].squeeze()
                            gt = self.totensor(gt).float()
            # TODO - delete
            if self.dataset == 'SHIFT':
                name = self.dataset_type[self.img_name][idx][self.dataset_type[self.img_name][idx].find('front') +6:self.dataset_type[self.img_name][idx].rfind('.jpg')]
            else:
                name = self.dataset_type[self.img_name][idx][self.dataset_type[self.img_name][idx].find('Data') +5:self.dataset_type[self.img_name][idx].rfind('.png')]        
            
            if self.sampler_input =='predict_from_past' or self.sampler_input =='None':
                return input, gt, past_depths, name
            
            elif self.sampler_input =='pseudo_gt':
                seg = 'empty'
                return input, gt, name, seg
            else:
                past_depths = ''
                return input, gt, past_depths
        
        elif self.past_inputs != 0 and self.sampler_input == 'predict_from_past':
            if self.dataset =='SHIFT':
                full_file_path = self.dataset_type[self.gt_name][idx]
                val_or_train = full_file_path[full_file_path.find('images')+7: full_file_path.find('front')- 1]
                file_name = full_file_path[full_file_path.rfind('/'):]
                
                file_index = str(int(file_name[1: file_name.find('_')])).rjust(8,'0')
                sceene_folder = full_file_path[full_file_path.find('front')+ 6: full_file_path.rfind('/')]
                indices_base_path = '/data/ashomer/project/SHIFT_dataset/pred_intime_depthmaps/'
                # predict_input = indices_base_path + val_or_train + '/' + sceene_folder + '/'+ file_index+'_img_front.npz'
                predict_input = indices_base_path  + '/' + sceene_folder + '/'+ file_index+'_img_front.npz'

                if os.path.exists(predict_input):
                    with np.load(predict_input, allow_pickle=True) as data:
                        tensor_predict_input = data['a'].squeeze()
                        tensor_predict_input = self.totensor(tensor_predict_input).float()
                else: 
                    raise Exception("No past data .npz file in {0}".format(predict_input))

                return input, gt, tensor_predict_input
            else:
                full_file_path = self.dataset_type[self.img_name][idx]
                base_pass_path = '/data/ashomer/project/SampleDepth/Data/pred_intime_depthmaps/'
                past_sample_data_path = base_pass_path + full_file_path[full_file_path.find('Data')+5:full_file_path.rfind(".png")]+".npz"
                if os.path.exists(past_sample_data_path):
                        # print(past_pseudogt_data_path)
                        with np.load(past_sample_data_path, allow_pickle=True) as data:
                            predict_input = data['a'].squeeze()
                            predict_input = self.totensor(predict_input).float()
                else: 
                    raise Exception("No past data .npz file in {0}".format(past_sample_data_path))

                pseudo_gt_base_path = '/data/ashomer/project/SampleDepth/Data/pseudo_gt/'
                past_pseudogt_data_path = pseudo_gt_base_path + full_file_path[full_file_path.find('Data')+5:full_file_path.rfind(".png")]+".npz"
                if os.path.exists(past_pseudogt_data_path):
                        # print(past_pseudogt_data_path)
                        with np.load(past_pseudogt_data_path, allow_pickle=True) as data:
                            gt = data['a'].squeeze()
                            gt = self.totensor(gt).float()
                seg = 'empty'

            return input, gt, predict_input, seg
                       
        else: 
            # TODO - delete
            if self.dataset == 'SHIFT':
                name = self.dataset_type[self.img_name][idx][self.dataset_type[self.img_name][idx].find('front') +6:self.dataset_type[self.img_name][idx].rfind('.jpg')]
            else:
                name = self.dataset_type[self.img_name][idx][self.dataset_type[self.img_name][idx].find('Data') +5:self.dataset_type[self.img_name][idx].rfind('.png')]
                seg = 'empty'
            return input, gt, name, seg
