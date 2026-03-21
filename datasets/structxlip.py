import logging

import torch
from torch.utils.data import DataLoader

from datasets.build import __factory, build_transforms
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from misc.structxlip_utils import PairStructTrainTransform, build_structure_caption, default_struct_lexicon_path, \
    load_struct_lexicon
from utils.comm import get_world_size
from utils.iotools import read_image

from .bases import ImageDataset, TextDataset, ImageTextMLMDataset, tokenize


class ImageTextStructDataset(ImageTextMLMDataset):
    def __init__(self, dataset, args, transform=None, text_length: int = 77, truncate: bool = True):
        super().__init__(dataset, transform=transform, text_length=text_length, truncate=truncate)
        self.args = args
        lexicon_path = args.struct_lexicon_file or str(default_struct_lexicon_path())
        self.lexicon = load_struct_lexicon(lexicon_path)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            images, edge_images = self.transform(img)
        else:
            images, edge_images = img, img

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        struct_caption = build_structure_caption(caption, self.args, self.lexicon)
        struct_caption_ids = tokenize(
            struct_caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate
        )

        ret = {
            "pids": pid,
            "image_ids": image_id,
            "images": images,
            "edge_images": edge_images,
            "caption_ids": caption_tokens,
            "mlm_ids": mlm_tokens,
            "mlm_labels": mlm_labels,
            "struct_caption_ids": struct_caption_ids,
        }
        return ret


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")
    return batch_tensor_dict


def build_dataloader_structxlip(args):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        pair_transform = PairStructTrainTransform(
            size=args.img_size,
            aug=args.img_aug,
            edge_threshold=args.struct_edge_threshold
        )
        val_transforms = build_transforms(img_size=args.img_size, is_train=False)

        train_set = ImageTextStructDataset(
            dataset.train,
            args=args,
            transform=pair_transform,
            text_length=args.text_length
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
                    collate_fn=collate
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
                    collate_fn=collate
                )
        elif args.sampler == "random":
            logger.info("using random sampler")
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate
            )
        else:
            logger.error(f"unsupported sampler: {args.sampler}")
            raise ValueError(f"unsupported sampler: {args.sampler}")

        ds = dataset.val if args.val_dataset == "val" else dataset.test
        val_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], val_transforms)
        val_txt_set = TextDataset(ds["caption_pids"], ds["captions"], text_length=args.text_length)

        val_img_loader = DataLoader(
            val_img_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        val_txt_loader = DataLoader(
            val_txt_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
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
        num_workers=num_workers
    )
    test_txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return test_img_loader, test_txt_loader, num_classes
