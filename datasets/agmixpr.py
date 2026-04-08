import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class AGMixPR(BaseDataset):
    """
    AGMix-PR dataset.

    The annotation format is close to CUHK-PEDES and stores image paths
    relative to the dataset root directory.
    """

    dataset_dir = 'AGMix_PR'

    def __init__(self, root='', verbose=True):
        super(AGMixPR, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir
        self.anno_path = op.join(self.dataset_dir, 'AGMix_PR.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info('=> AGMix-PR Images and Captions are loaded')
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            split = anno['split']
            if split == 'train':
                train_annos.append(anno)
            elif split == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            raw_pids = sorted({int(anno['id']) for anno in annos})
            pid2label = {pid: idx for idx, pid in enumerate(raw_pids)}
            dataset = []
            image_id = 0
            for anno in annos:
                raw_pid = int(anno['id'])
                pid = pid2label[raw_pid]
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                for caption in anno['captions']:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            return dataset, pid_container

        img_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        for anno in annos:
            pid = int(anno['id'])
            pid_container.add(pid)
            img_path = op.join(self.img_dir, anno['file_path'])
            img_paths.append(img_path)
            image_pids.append(pid)
            for caption in anno['captions']:
                captions.append(caption)
                caption_pids.append(pid)
        dataset = {
            'image_pids': image_pids,
            'img_paths': img_paths,
            'caption_pids': caption_pids,
            'captions': captions,
        }
        return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
