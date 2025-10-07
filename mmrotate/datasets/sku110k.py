import glob
import os.path as osp
from typing import List

#from .dota import DOTADataset
from mmrotate.registry import DATASETS
from mmengine.dataset import BaseDataset
import json

@DATASETS.register_module()
class SKU110KDataset(BaseDataset):
    METAINFO = {
        'classes':
        (
            'object',
        ),
        'palette':
        [
            (165, 42, 42)
        ]
    }
    
    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',   #sku
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {#'object':1
                   c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

        elif self.ann_file.endswith('.json'):   #sku
            with open(self.ann_file, 'r') as f:
                root = json.loads(f.read())

            instances = {}
            for item in root:
                img_id = item['image_id']
                if img_id not in instances.keys():
                    instances[img_id] = []
                instances[img_id].append({'bbox': item['rbbox'] if 'rbbox' in item.keys() else item['bbox'],
                                          'bbox_label': 0,
                                          'ignore_flag': 0})

            for img_id in instances.keys():
                data_info = {}
                data_info['img_id'] = img_id
                img_name = str(img_id) + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)
                data_info['instances'] = instances[img_id]
                data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
