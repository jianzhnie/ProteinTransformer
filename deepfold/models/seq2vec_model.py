import logging
import urllib.request
from pathlib import Path

from allennlp.modules.elmo import Elmo

logger = logging.getLogger(__name__)


def get_elmo_model(model_dir: Path) -> Elmo:
    weights_path = model_dir / 'weights.hdf5'
    options_path = model_dir / 'options.json'

    # if no pre-trained model is available, yet --> download it
    if not (weights_path.exists() and options_path.exists()):
        logger.info(
            'No existing model found. Start downloading pre-trained SeqVec (~360MB)...'
        )

        Path.mkdir(model_dir, exist_ok=True)
        repo_link = 'http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec'
        options_link = repo_link + '/options.json'
        weights_link = repo_link + '/weights.hdf5'
        urllib.request.urlretrieve(options_link, str(options_path))
        urllib.request.urlretrieve(weights_link, str(weights_path))

    logger.info('Loading the model')
    # The string casting comes from a typing bug in allennlp
    # https://github.com/allenai/allennlp/pull/3358
    return Elmo(weight_file=weights_path,
                options_file=options_path,
                num_output_representations=3)


if __name__ == '__main__':
    model_dir = Path('.')
    model = get_elmo_model(model_dir)
    print(model)
