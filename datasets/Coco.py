import os.path as op
import re
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class Coco(BaseDataset):
    dataset_dir = 'coco'

    def __init__(self, root='', verbose=True):
        super(Coco, self).__init__()
        self.dataset_dir = ' '
        self.img_train_dir = op.join(self.dataset_dir, 'train2014/')
        self.img_test_dir = op.join(self.dataset_dir, 'val2014/')

        filenames = { 'train': ' ',
                'val': ' ',
                'test': ' '}

        self.anno_path = filenames
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> coco Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos = read_json(anno_path['train'])
        test_annos = read_json(anno_path['test'])
        val_annos = read_json(anno_path['val'])
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            for anno in annos:
                match = re.search(r'\d+', anno['image_id'])
                if match:
                    extracted_number = int(match.group())
                else:
                    print("No number found in the string.")

                pid = extracted_number
                pid_container.add(pid)
                image_id = extracted_number
                img_path = op.join(self.dataset_dir, anno['image'])
                captions = anno['caption']  # caption list
                dataset.append((pid, image_id, img_path, captions))
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for img_id, anno in enumerate(annos):
                pid = int(img_id)
                pid_container.add(pid)
                img_path = op.join(self.dataset_dir,  anno['image'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['caption']  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_train_dir):
            raise RuntimeError("'{}' is not available".format(self.img_train_dir))
        if not op.exists(self.img_test_dir):
            raise RuntimeError("'{}' is not available".format(self.img_test_dir))
        if not op.exists(self.anno_path['train']):
            raise RuntimeError("'{}' is not available".format(self.anno_path['train']))
        if not op.exists(self.anno_path['test']):
            raise RuntimeError("'{}' is not available".format(self.anno_path['test']))
        if not op.exists(self.anno_path['val']):
            raise RuntimeError("'{}' is not available".format(self.anno_path['val']))
