from .deepgoplus import DeepGOPlusModel
from .esm_model import EsmEmbeddingModel, EsmTransformer
from .modeling_lstm import (ContrastiveProteinLSTMModel,
                            MultilabelProteinLSTMModel, ProteinLSTMConfig)


def get_model(args):
    if args.model == 'deepgoplus':
        model = DeepGOPlusModel(args.num_labels)

    if args.model == 'esm':

        model = EsmTransformer(num_labels=args.num_labels,
                               pool_mode=args.pool_mode,
                               fintune=False)
    if args.model == 'esm_embedding':

        model = EsmEmbeddingModel(input_size=1280, num_labels=args.num_labels)

    if args.model == 'lstm':

        config = ProteinLSTMConfig(num_labels=args.num_labels)
        model = MultilabelProteinLSTMModel(config)

    if args.model == 'contrastive_lstm':

        config = ProteinLSTMConfig()
        model = ContrastiveProteinLSTMModel(config)

    return model
