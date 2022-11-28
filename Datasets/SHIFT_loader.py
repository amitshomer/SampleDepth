"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re
import numpy as np
from PIL import Image
import imghdr
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from Utils.utils import write_file, depth_read
'''
attention:
    There is mistake in 2011_09_26_drive_0009_sync/proj_depth 4 files were
    left out 177-180 .png. Hence these files were also deleted in rgb
'''




class SHIFT_preprocessing(object):
    def __init__(self, dataset_path, input_type='depth', side_selection=''):
        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}
        self.dataset_path = dataset_path
        self.use_rgb = True

    def get_paths(self, past_inputs = 0):
        # train and validation dirs
        if past_inputs == 0:
            remove_list_rgb = []
            remove_list_depth = []

        elif past_inputs == 1:
            remove_list_rgb = ['00000000_img_front.jpg']
            remove_list_depth = ['00000000_depth_front.png']

        elif past_inputs == 2: 
            remove_list_rgb = ['00000000_img_front.jpg','00000010_img_front.jpg']
            remove_list_depth = ['00000000_depth_front.png','00000010_depth_front.png' ]
        
        elif past_inputs == 3: 
            remove_list_rgb = ['00000000_img_front.jpg','00000010_img_front.jpg','00000020_img_front.jpg']
            remove_list_depth = ['00000000_depth_front.png','00000010_depth_front.png','00000020_depth_front.png' ]
        
        elif past_inputs == 4: 
            remove_list_rgb = ['00000000_img_front.jpg','00000010_img_front.jpg','00000020_img_front.jpg','00000030_img_front.jpg']
            remove_list_depth = ['00000000_depth_front.png','00000010_depth_front.png','00000020_depth_front.png','00000030_depth_front.png' ]
        
        elif past_inputs == 5: 
            remove_list_rgb = ['00000000_img_front.jpg','00000010_img_front.jpg','00000020_img_front.jpg','00000030_img_front.jpg','00000040_img_front.jpg']
            remove_list_depth = ['00000000_depth_front.png','00000010_depth_front.png','00000020_depth_front.png','00000030_depth_front.png','00000040_depth_front.png' ]

        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                    if re.search('jpg', file)
                                                    and re.search('train', root)
                                                    and file not in remove_list_rgb]))
                
                self.train_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                    if re.search('png', file)
                                    and re.search('train', root)
                                    and file not in remove_list_depth]))
                
                self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                    if re.search('jpg', file)
                                    and re.search('val', root)
                                    and file not in remove_list_rgb]))
                
                self.val_paths['gt'].extend(sorted([os.path.join(root, file) for file in files
                                    if re.search('png', file)
                                    and re.search('val', root)
                                    and file not in remove_list_depth]))
                
                
    def prepare_dataset(self, past_inputs =0, plot_paper = False, sampler_input='gt'):
        self.get_paths(past_inputs = past_inputs)
        print(len(self.train_paths['lidar_in']))
        print(len(self.train_paths['img']))
        print(len(self.train_paths['gt']))
        print(len(self.val_paths['lidar_in']))
        print(len(self.val_paths['img']))
        print(len(self.val_paths['gt']))

        if plot_paper: 
            # if sampler_input =='gt':
            #     indices = [5,1000,4500,9400,14300,16700,19110,20000,935,13900]
            # else:
            #     indices = [1,1000,4500,9400,14300,16700,19110,20000,935,13900]

            list_img = [
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/4056-844f/00000050_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/8b25-a278/00000310_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/4a92-8eb3/00000120_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/0e0c-b33d/00000160_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/ef7b-f9c3/00000200_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/1ae5-2eb5/00000230_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/dfca-6637/00000080_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/2eb2-0deb/00000370_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/d4d7-ff2c/00000390_img_front.jpg',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/8d04-4174/00000310_img_front.jpg',

            ]
            list_gt = [
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/4056-844f/00000050_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/8b25-a278/00000310_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/4a92-8eb3/00000120_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/0e0c-b33d/00000160_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/ef7b-f9c3/00000200_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/1ae5-2eb5/00000230_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/dfca-6637/00000080_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/2eb2-0deb/00000370_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/d4d7-ff2c/00000390_depth_front.png',
                '/data/ashomer/project/SHIFT_dataset/discrete/images/val/front/8d04-4174/00000310_depth_front.png',
            ]
            self.val_paths['img'] = list_img
            self.val_paths['gt'] = list_gt
            print("plot_paper data")
    def compute_mean_std(self):
        nums = np.array([])
        means = np.array([])
        stds = np.array([])
        max_lst = np.array([])
        for i, raw_img_path in tqdm.tqdm(enumerate(self.train_paths['lidar_in'])):
            raw_img = Image.open(raw_img_path)
            raw_np = depth_read(raw_img)
            vec = raw_np[raw_np >= 0]
            # vec = vec/84.0
            means = np.append(means, np.mean(vec))
            stds = np.append(stds, np.std(vec))
            nums = np.append(nums, len(vec))
            max_lst = np.append(max_lst, np.max(vec))
        mean = np.dot(nums, means)/np.sum(nums)
        std = np.sqrt((np.dot(nums, stds**2) + np.dot(nums, (means-mean)**2))/np.sum(nums))
        return mean, std, max_lst


if __name__ == '__main__':

    # Imports
    import tqdm
    from PIL import Image
    import os
    import argparse
    from Utils.utils import str2bool

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument("--png2img", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--calc_params", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    parser.add_argument('--datapath', default='/usr/data/tmp/Depth_Completion/data')
    parser.add_argument('--dest', default='/usr/data/tmp/')
    args = parser.parse_args()

    dataset = Kitti_preprocessing(args.datapath, input_type='rgb')
    dataset.prepare_dataset()
    if args.png2img:
        os.makedirs(os.path.join(args.dest, 'Rgb'), exist_ok=True)
        destination_train = os.path.join(args.dest, 'Rgb/train')
        destination_valid = os.path.join(args.dest, 'Rgb/val')
        dataset.convert_png_to_rgb(dataset.train_paths['img'], destination_train)
        dataset.convert_png_to_rgb(dataset.val_paths['img'], destination_valid)
    if args.calc_params:
        import matplotlib.pyplot as plt
        params = dataset.compute_mean_std()
        mu_std = params[0:2]
        max_lst = params[-1]
        print('Means and std equals {} and {}'.format(*mu_std))
        plt.hist(max_lst, bins='auto')
        plt.title('Histogram for max depth')
        plt.show()
        # mean, std = 14.969576188369581, 11.149000139428104
        # Normalized
        # mean, std = 0.17820924033773314, 0.1327261921360489
    if args.num_samples != 0:
        print("Making downsampled dataset")
        os.makedirs(os.path.join(args.dest), exist_ok=True)
        destination_train = os.path.join(args.dest, 'train')
        destination_valid = os.path.join(args.dest, 'val')
        dataset.downsample(dataset.train_paths['lidar_in'], destination_train, args.num_samples)
        dataset.downsample(dataset.val_paths['lidar_in'], destination_valid, args.num_samples)
