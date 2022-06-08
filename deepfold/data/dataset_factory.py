import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from .esm_dataset import EsmDataset, EsmEmbeddingDataset
from .protein_dataset import ProtBertDataset, ProtSeqDataset


def get_dataloaders(args):
    name = args.dataset_name.lower()
    if name == 'esm':
        train_dataset = EsmDataset(data_path=args.data_path, split='train')
        val_dataset = EsmDataset(data_path=args.data_path, split='test')
    elif name == 'esm_embedding':
        train_dataset = EsmEmbeddingDataset(data_path=args.data_path,
                                            split='train')
        val_dataset = EsmEmbeddingDataset(data_path=args.data_path,
                                          split='test')
    elif name == 'protseq':
        train_dataset = ProtSeqDataset(data_path=args.data_path, split='train')
        val_dataset = ProtSeqDataset(data_path=args.data_path, split='test')
    elif name == 'bert':
        train_dataset = ProtBertDataset(data_path=args.data_path,
                                        split='train')
        val_dataset = ProtBertDataset(data_path=args.data_path, split='test')
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.RandomSampler(val_dataset)

    if name in ['esm', 'protseq']:
        collate_fn = train_dataset.collate_fn
    else:
        collate_fn = None

    # dataloders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        collate_fn=collate_fn,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        collate_fn=collate_fn,
        sampler=val_sampler,
        pin_memory=True,
    )
    return train_loader, val_loader
