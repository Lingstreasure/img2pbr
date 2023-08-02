import os
from typing import Any, List, Dict

import numpy as np
import hydra
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PBROvercastHdrRenderDataset(Dataset):
    def __init__(
        self,
        data_root_dir: str,
        data_list_file_dir: str, 
        subdataset_names: list,
        pbr_dir_names: list,
        rendered_dir_names: list,
        mode: str = "train",
        pbr_maps_type: str = "all",
        img_transforms: list = [],
        debug_on: bool = False,
        debug_data_num: int = 100, 
    ) -> None:
        """Initialize a `PBROvercastHdrRenderDataset`.

        :param data_root_dir: The root directory of data.
        :param data_list_file_dir: The directory of data list file, used to select samples.
        :param subdataset_names: The subdatasets to be used in [3dtextures, ambient, polyhaven, sharetextures].
        :param pbr_dir_names: The pbr maps directory of subdatasets to be used.
        :param render_dir_names: The rendered imgs directory of subdatasets to be used.
        :param mode: The training mode of this dataset used for, train or test.
        :param pbr_maps_type: The pbr types to be used. Default to `all`.
        :param img_transforms: The img transformations for data augmentation.
        :param debug_on: Build the dataset for debugging or testing.
        :param debug_data_num: The number of samples for debugging or testing. Only effective when `debug_on=True`.
        """
        super().__init__()
        self.data_root_dir = data_root_dir
        self.data_list_file_dir = data_list_file_dir
        self.subdataset_names = subdataset_names
        self.pbr_dir_names = pbr_dir_names
        self.rendered_dir_names = rendered_dir_names
        self.mode = mode
        self.pbr_maps_type = pbr_maps_type
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
        
        # load data of different subdatasets with specific data_list_file
        for dataset_name, pbr_dir_name, rendered_dir_name in zip(
            self.subdataset_names, self.pbr_dir_names, self.rendered_dir_names
        ):
            data_list_file_path = os.path.join(
                self.data_list_file_dir, dataset_name + '_' + self.mode + '.txt'
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
            # load all pbr maps and rendered imgs with sample name list
            for name in sample_names:
                # debug with a subset of whole datatset
                if len(data) >= self.debug_data_num:
                    break
                
                rendered_path = os.path.join(
                    self.data_root_dir, rendered_dir_name, name + "_512.png"
                )
                if not os.path.isfile(rendered_path):
                    continue
                
                gt_paths = {}
                pbr_sample_dir = os.path.join(self.data_root_dir, pbr_dir_name, name)
                gt_paths['albedo'] = os.path.join(pbr_sample_dir, 'color_512.jpg')
                gt_paths['normal'] = os.path.join(pbr_sample_dir, 'normaldx_512.jpg')
                gt_paths['roughness'] = os.path.join(pbr_sample_dir, 'roughness_512.jpg')
                gt_paths['metalness'] = os.path.join(pbr_sample_dir, 'metallic_512.jpg')
                
                isValid = True
                # check each pbr map path except metal.
                for key, path in gt_paths.items():
                    if key == 'metalness':
                        continue
                    elif not os.path.isfile(path):
                        isValid = False
                        break
                if not isValid:
                    continue
                
                # load imgs
                input = Image.open(rendered_path)  # PIL.Image，h w 3
                input = np.asarray(input.convert("RGB"))
                
                gts = {}
                color = Image.open(gt_paths['albedo'])  # h w 3
                color = np.asarray(color.convert("RGB"))
                gts['albedo'] = color

                normal = Image.open(gt_paths['normal'])  # h w 3
                normal = np.asarray(normal.convert("RGB"))
                gts['normal'] = normal
                
                roughness = Image.open(gt_paths['roughness'])  # h w 3
                roughness = np.asarray(roughness.convert("RGB"))
                gts['roughness'] = roughness[..., 0, None]  # h w 3 -> h w 1
                
                metalness = np.zeros((512, 512, 1)).astype(np.uint8)
                if os.path.exists(gt_paths['metalness']):
                    metalness = Image.open(gt_paths['metalness'])
                    metalness = np.asarray(metalness.convert("RGB"))
                gts['metalness'] = metalness[..., 0, None]  # h w 3 -> h w 1

                gt = np.concatenate([gts['albedo'], gts['normal'], gts["roughness"], gts["metalness"]], axis=-1)
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
    
    def __getitem__(self, index) -> Dict:
        name, input, gt = self._data[index]
        input = self._img_process(input)
        gt = self._img_process(gt)
        return {"name": name, "input": input, "gt": gt}
    

if __name__ == "__main__":
    dataset = PBROvercastHdrRenderDataset(
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