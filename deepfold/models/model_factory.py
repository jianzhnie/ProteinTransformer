from deepgoplus import DeepGOPlusModel
from esm_model import EsmEmbeddingModel, EsmTransformer
from modeling_lstm import MultilabelProteinLSTMModel, ProteinLSTMConfig


def get_model(args):
    model_name = args.model_name

    if model_name == 'deepgoplus':
        model = DeepGOPlusModel(args.num_labels)

    if model_name == 'esm':

        model = EsmTransformer(num_labels=args.num_labels,
                               pool_mode=args.pool_mode,
                               fintune=False)
    if model_name == 'esm_embedding':

        model = EsmEmbeddingModel(input_size=1280, num_labels=args.num_labels)

    if model_name == 'lstm':

        config = ProteinLSTMConfig(num_labels=args.num_labels)
        model = MultilabelProteinLSTMModel(config)

    return model
