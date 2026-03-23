import logging

import torch
from torch.utils.data import DataLoader

from datasets.build import __factory, build_transforms, collate
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from misc.color_utils import build_color_label, default_color_lexicon_path, load_color_lexicon
from utils.comm import get_world_size

from .bases import ImageDataset, ImageTextMLMDataset, TextDataset


class ImageTextColorDataset(ImageTextMLMDataset):
    def __init__(self, dataset, args, transform=None, text_length=77, truncate=True):
        super().__init__(dataset, transform=transform, text_length=text_length, truncate=truncate)
        self.use_mlm = args.MLM
        lexicon_path = args.color_lexicon_file or str(default_color_lexicon_path())
        self.lexicon = load_color_lexicon(lexicon_path)

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if not self.use_mlm:
            ret.pop("mlm_ids", None)
            ret.pop("mlm_labels", None)
        _, _, _, caption = self.dataset[index]
        color_labels = build_color_label(caption, self.lexicon)
        ret["color_labels"] = color_labels
        ret["has_color_label"] = int(color_labels.sum().item() > 0)
        return ret


def build_dataloader_color(args):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    val_num_workers = args.val_num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        train_transforms = build_transforms(
            img_size=args.img_size,
            aug=args.img_aug,
            is_train=True,
        )
        val_transforms = build_transforms(
            img_size=args.img_size,
            is_train=False,
        )

        train_set = ImageTextColorDataset(
            dataset.train,
            args=args,
            transform=train_transforms,
            text_length=args.text_length,
        )

        if args.sampler == "identity":
            if args.distributed:
                logger.info("using ddp random identity sampler")
                logger.info("DISTRIBUTED TRAIN START")
                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader = DataLoader(
                    train_set,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    collate_fn=collate,
                )
            else:
                logger.info(
                    f"using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}"
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance),
                    num_workers=num_workers,
                    collate_fn=collate,
                )
        elif args.sampler == "random":
            logger.info("using random sampler")
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate,
            )
        else:
            logger.error(f"unsupported sampler: {args.sampler}")
            raise ValueError(f"unsupported sampler: {args.sampler}")

        ds = dataset.val if args.val_dataset == "val" else dataset.test
        val_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], val_transforms)
        val_txt_set = TextDataset(ds["caption_pids"], ds["captions"], text_length=args.text_length)

        val_img_loader = DataLoader(
            val_img_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
        )
        val_txt_loader = DataLoader(
            val_txt_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
        )

        return train_loader, val_img_loader, val_txt_loader, num_classes

    test_transforms = build_transforms(img_size=args.img_size, is_train=False)
    ds = dataset.test
    test_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], test_transforms)
    test_txt_set = TextDataset(ds["caption_pids"], ds["captions"], text_length=args.text_length)

    test_img_loader = DataLoader(
        test_img_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
    )
    test_txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
    )
    return test_img_loader, test_txt_loader, num_classes
