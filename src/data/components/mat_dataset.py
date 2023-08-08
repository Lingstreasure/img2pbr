import os
from typing import Any, List, Tuple

import hydra
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MatMatchingDataset(Dataset):
    """A material matching(aka. classification) dataset."""
    
    def __init__(
        self,
        data_root_dir: str,
        data_list_file_dir: str, 
        list_file_prefix: str,
        mode: str = "train",
        img_transforms: list = [],
        debug_on: bool = False,
        debug_data_num: int = 100, 
    ) -> None:
        """Initialize a `MatMatchingDataset`.

        :param data_root_dir: The root directory of data.
        :param data_list_file_dir: The directory of data list file, used to select samples.
        :param list_file_prefix: The prefix of material data list file.
        :param mode: The training mode of this dataset used for, train or test.
        :param img_transforms: The img transformations for data augmentation.
        :param debug_on: Build the dataset for debugging or testing.
        :param debug_data_num: The number of samples for debugging or testing. Only effective when `debug_on=True`.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.data_list_file_dir = data_list_file_dir
        self.list_file_prefix = list_file_prefix
        self.mode = mode
        self.debug_on = debug_on
        self.debug_data_num = debug_data_num if debug_on else 1000000000
        self._tform = self._set_transforms(img_transforms)
        self._data = self._preprocess()
        
    def __len__(self) -> int:
        """Get the number of samples in dataset.
        :return: The number of samples in dataset.
        """
        return len(self._data)
        
    def _preprocess(self) -> List[List[Any]]:
        """Process the raw data and match different pbr maps togather.
        :return: The list of processed samples.
        """
        # list to be returned
        data = []
        
        # load data with specific data_list_file
        data_list_file_path = os.path.join(
            self.data_root_dir, self.list_file_prefix + '_' + self.mode + '.txt'
        )
        # sample name list
        sample_names = []
        with open(data_list_file_path) as f:
            lines = f.readlines()
            for line in lines:
                name = line.strip()
                sample_names.append(name)
        assert len(sample_names) > 0, "No data here"

        cnt = 0
        # load all material images to be classified
        for name in sample_names:
            # debug with a subset of whole datatset
            if len(data) >= self.debug_data_num:
                break
            
            img_path = os.path.join(self.data_root_dir, name + "_512.png")
            label_path = os.path.join(self.data_root_dir, name + "_label.txt")
            if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
                continue
            
            # load input img and it's class label
            input = Image.open(img_path)  # PIL.Image，h w 3
            input = np.asarray(input.convert("RGB"))
            with open(label_path, 'r') as f:
                gt = int(f.read().strip())
                
            data.append([name, input, gt])
            cnt += 1

        return data
    
    def _set_transforms(self, img_transforms: list = [transforms, ...]) -> transforms:
        """Set the transforms, such as resize(), ToTensor(), for imgs. 
        :return: The composed img transforms in torchvision.tranforms.
        """
        img_transforms = [hydra.utils.instantiate(tt) for tt in img_transforms]
        img_transforms.extend([
            transforms.ToTensor(),  # row_data -> (0, 1)
            transforms.Lambda(lambda x: x * 2. - 1.)  # (0, 1) -> (-1, 1)
        ])
        return transforms.Compose(img_transforms)
    
    def _img_process(self, image: Image) -> Image:
        """Apply the transforms to imgs before entering the model.
        :return: The transformed imgs.
        """
        return self._tform(image)
    
    def __getitem__(self, index) -> Tuple[Any]:
        name, input, gt = self._data[index]
        input = self._img_process(input)
        gt = torch.tensor(gt, dtype=torch.int32)
        return (name, input, gt)
    

if __name__ == "__main__":
    dataset = MatMatchingDataset(
        data_root_dir="/media/d5/7D1922F98D178B12/dw/mat_reorg",
        data_list_file_dir="/home/d5/hz/Code/sdmat/data_file", 
        subdataset_names=['3dtextures', 'ambient', 'polyhaven', 'sharetextures'],
        pbr_dir_names=['3dtextures_cleanup_nometal', 'ambient_cleanup_nometal', 'polyhaven_cleanup', 'sharetextures_cleanup_nometal'],
        rendered_dir_names=['3dtextures_renderout_blender', 'ambient_renderout_blender', 'polyhaven_renderout_blender', 'sharetextures_renderout_blender'],
        mode='train',
        pbr_maps_type='all',
        img_transforms=[],
        debug_on=True
    )
    print(len(dataset))