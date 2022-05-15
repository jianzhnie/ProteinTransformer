import logging
import os
import shutil

import torch
from torch import distributed as dist

_logger = logging.getLogger(__name__)


def save_checkpoint(state,
                    is_best,
                    checkpoint_dir,
                    filename='checkpoint.pth.tar'):
    if (not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
        file_path = os.path.join(checkpoint_dir, filename)
        torch.save(state, file_path)
        if is_best:
            shutil.copyfile(file_path,
                            os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (torch.distributed.get_world_size()
           if torch.distributed.is_initialized() else 1)
    return rt


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs."""
    lr = args.lr * (0.1**(epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_model_checkpoint(args):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(
            args.resume,
            map_location=lambda storage, loc: storage.cuda(args.gpu))
        checkpoint = {
            k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in checkpoint.items()
        }
        optimizer_state = checkpoint['optimizer']
        model_state = checkpoint['state_dict']
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        model_state = None
        optimizer_state = None

    return model_state, optimizer_state
