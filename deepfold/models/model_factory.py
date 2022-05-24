from deepfold.data.esm_dataset import ESMDataset
from deepfold.models.esm_model import EsmTransformer
from deepfold.data.esm_dataset import EsmEmbeddingDataset
from deepfold.models.esm_model import EsmEmbeddingModel


def model_factory(model_name, embedding=True):

    return 