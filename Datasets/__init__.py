
from .Kitti_loader import *
from .SHIFT_loader import *

# dataset_dict = {'kitti': Kitti_preprocessing}

def allowed_datasets(data):
    if data =='kitti':
        dataset_dict = {'kitti': Kitti_preprocessing}
    else:
        dataset_dict = {'SHIFT': SHIFT_preprocessing}

    return dataset_dict.keys()

def define_dataset(data, *args):
    if data =='kitti':
        dataset_dict = {'kitti': Kitti_preprocessing}
        name = 'kitti'
    else:
        dataset_dict = {'SHIFT': SHIFT_preprocessing}
        name = 'SHIFT'

    if data not in allowed_datasets(data):
        raise KeyError("The requested dataset is not implemented")
    else:
        return dataset_dict[name](*args)
        
