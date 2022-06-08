from .esm_dataset import EsmDataset, EsmEmbeddingDataset
from .protein_dataset import CustomProtSeqDataset, ProtBertDataset


def get_dataset(args):
    name = args.dataset_name.lower()
    if name == 'esm':
        train_dataset = EsmDataset(data_path=args.data_path, split='train')
        test_dataset = EsmDataset(data_path=args.data_path, split='test')
    elif name == 'esm_embedding':
        train_dataset = EsmEmbeddingDataset(data_path=args.data_path,
                                            split='train')
        test_dataset = EsmEmbeddingDataset(data_path=args.data_path,
                                           split='test')
    elif name == 'custom':
        train_dataset = CustomProtSeqDataset(data_path=args.data_path,
                                             split='train')
        test_dataset = CustomProtSeqDataset(data_path=args.data_path,
                                            split='test')
    elif name == 'prot_bert':
        train_dataset = ProtBertDataset(data_path=args.data_path,
                                        split='train')
        test_dataset = ProtBertDataset(data_path=args.data_path, split='test')

    else:
        raise NotImplementedError
    return train_dataset, test_dataset
