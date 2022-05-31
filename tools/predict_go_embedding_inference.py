import argparse

from deepfold.gosim.function_prediction import FunctionPrediction
from deepfold.gosim.gene_ontology import GeneOntology
from deepfold.utils.file_utils import read_embeddings, read_go_annotations

parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--train_embedding_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--test_embedding_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--go_file', default='go.obo', help='go file')
parser.add_argument('--annotations',
                    default='go_annotations',
                    help='go annotations')
parser.add_argument('--onto', default='all', type=str, help='set ontologies')


def main(args):
    # read in embeddings, annotations, and GO
    test_embeddings = read_embeddings(args.test_embedding_path)
    embeddings = read_embeddings(args.train_embedding_path)

    go = GeneOntology(args.go_file)
    go_annotations = read_go_annotations(args['annotations'])

    # set ontologies
    if args.onto == 'all':
        ontologies = ['bpo', 'mfo', 'cco']
    else:
        ontologies = args.onto

    # set dist cutoffs:
    cutoffs = args.thresh
    dist_cutoffs = cutoffs.split(',')

    # perform prediction for each ontology individually
    for go_sub in ontologies:
        predictor = FunctionPrediction(embeddings, go_annotations, go, go_sub)
        predictions_all, _ = predictor.run_prediction_embedding_all(
            test_embeddings, 'euclidean', dist_cutoffs, args.modus)

        # write predictions for each distance cutoff
        for dist in predictions_all.keys():
            predictions = predictions_all[dist]
            predictions_out = '{}_{}_{}.txt'.format(args['output'], dist,
                                                    go_sub)
            FunctionPrediction.write_predictions(predictions, predictions_out)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
