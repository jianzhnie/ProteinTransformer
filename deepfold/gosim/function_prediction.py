import sys
from collections import defaultdict

import numpy
from embedding_lookup import EmbeddingLookup


class FunctionPrediction(object):
    def __init__(self, embedding_db, go_annotation, gene_ontology, go_type):
        self.gen_ontology = gene_ontology

        if go_type == 'all':
            self.embedding_lookup = EmbeddingLookup(embedding_db)
            self.go_annotation = go_annotation
        elif go_type == 'mfo' or go_type == 'bpo' or go_type == 'cco':
            # only use proteins in the annotation set which actually have an annotation in this ontology
            self.go_annotation = defaultdict(set)
            embedding_db_reduced = dict()
            for k in embedding_db.keys():
                terms = go_annotation[k]
                go_terms = self.get_terms_by_go(terms)[go_type]
                if len(go_terms) > 0:
                    embedding_db_reduced[k] = embedding_db[k]
                    self.go_annotation[k] = go_terms
            self.embedding_lookup = EmbeddingLookup(embedding_db_reduced)
        else:
            sys.exit(
                '{} is not a valid GO. Valid GOs are [all|mfo|bpo|cco]'.format(
                    go_type))

    def get_terms_by_go(self, terms):
        terms_by_go = {'mfo': set(), 'bpo': set(), 'cco': set()}

        for t in terms:
            onto = self.gene_ontology.get_ontology(t)
            if onto != '':
                terms_by_go[onto].add(t)

        return terms_by_go

    def run_prediction_embedding_all(self, querys, distance, hits, criterion):
        """Perform inference based on embedding-similarity.

        :param querys: proteins for which GO terms should be predicted
        :param distance: distance measure to use [euclidean|cosine]
        :param hits: hits to include (either by distance or by number as defined with criterion)
        :param criterion: should k closest hits or all hits with distance <k be included?
        :return:
        """

        predictions = defaultdict(defaultdict)
        hit_ids = defaultdict(defaultdict)

        distances, query_ids = self.embedding_lookup.run_embedding_lookup_distance(
            querys, distance)

        for i in range(0, len(query_ids)):
            query = query_ids[i]
            dists = distances[i, :].squeeze()
            for h in hits:
                prediction = dict()
                if criterion == 'dist':  # extract hits within a certain distance
                    h = float(h)
                    indices = numpy.nonzero(dists <= h)
                elif criterion == 'num':  # extract h closest hits
                    h = int(h)
                    indices_tmp = numpy.argpartition(dists, h)[0:h]
                    dists_tmp = [dists[i] for i in indices_tmp]
                    max_dist = numpy.amax(dists_tmp)
                    indices = numpy.nonzero(dists <= max_dist)[0]

                    if len(indices) > h:
                        print(
                            'Multiple hits with same distance found, resulting in {} hits'
                            .format(len(indices)))

                else:
                    sys.exit(
                        'No valid criterion defined, valid criterions are [dist|num]'
                    )

                num_hits = len(indices)

                #  1. 对每个 qury 蛋白, 在数据库中找到符合标准的 K 个相似蛋白
                #       2. 对 K 个相似蛋白， 根据找到的蛋白id 获取 Go term annotation 及 对应的距离
                #       3. 将距离标准化为x相似性得分
                #       4. 预测结果输出 :
                #                     {'GO:001: 0.878',
                #                       'GO:002: 0.8'}
                for ind in indices:
                    lookup_id = self.embedding_lookup.ids[ind]
                    go_terms = self.go_annotation[lookup_id]
                    dist = dists[ind]

                    if distance == 'euclidean':
                        # scale distance to reflect a similarity [0;1]
                        dist = 0.5 / (0.5 + dist)
                    elif distance == 'cosine':
                        dist = 1 - dist

                    for g in go_terms:
                        if g in prediction.keys():
                            # if multiple hits are included RIs get smaller --> predictions retrieved for different
                            # numbers of hits are not directly comparable
                            prediction[g] += dist / num_hits
                        else:
                            prediction[g] = dist / num_hits

                    if query not in hit_ids[h].keys():
                        hit_ids[h][query] = dict()
                    hit_ids[h][query][lookup_id] = round(dist, 2)

                # round ri and remove hits with ri == 0.00
                keys_for_deletion = set()
                for p in prediction:
                    ri = round(prediction[p], 2)
                    if ri == 0.00:
                        keys_for_deletion.add(p)
                    else:
                        prediction[p] = ri

                for k in keys_for_deletion:
                    del prediction[k]

                # reduce prediction to leaf terms
                parent_terms = []
                for p in prediction.keys():
                    parents = self.gen_ontology.get_parent_terms(p)
                    parent_terms += parents
                # exclude terms that are parent terms, i.e. there are more specific terms also part of this prediction
                keys_for_deletion = set()
                for p in prediction.keys():
                    if p in parent_terms:
                        keys_for_deletion.add(p)

                for k in keys_for_deletion:
                    del prediction[k]

                predictions[h][query] = prediction

        return predictions, hit_ids

    def run_prediction_one_target(self, query_embedding, distance, k,
                                  criterion):
        """Perform inference based on embedding-similarity for one query
        embedding.

        :param query_embedding: query to calculate prediction for
        :param distance: distance measure to use [euclidean|cosine]
        :param k: hits to include (either by distance or by number as defined with criterion)
        :param criterion: Should k closest hits or all hits with distance <k be included?
        :return: GO term predictions with RI
        """

        prediction = dict()
        distances, _ = self.embedding_lookup.run_embedding_lookup_distance(
            query_embedding, distance)
        dists = distances[0, :].squeeze().numpy()

        if criterion == 'dist':  # extract hits within a certain distance
            k = float(k)
            indices = numpy.nonzero(dists <= k)
        elif criterion == 'num':  # extract h closest hits
            k = int(k)
            indices_tmp = numpy.argpartition(dists, k)[0:k]
            dists_tmp = [dists[i] for i in indices_tmp]
            max_dist = numpy.amax(dists_tmp)
            indices = numpy.nonzero(dists <= max_dist)[0]
        else:
            sys.exit(
                'No valid criterion defined, valid criterions are [dist|num]')

        num_hits = len(indices)

        for ind in indices:
            lookup_id = self.embedding_lookup.ids[ind]
            go_terms = self.go_annotation[lookup_id]
            dist = dists[ind]

            if distance == 'euclidean':
                # scale distance to reflect a similarity [0;1]
                dist = 0.5 / (0.5 + dist)
            elif distance == 'cosine':
                dist = 1 - dist

            for g in go_terms:
                if g in prediction.keys():
                    # if multiple hits are included RIs get smaller --> predictions retrieved for different
                    # numbers of hits are not directly comparable
                    prediction[g] += dist / num_hits
                else:
                    prediction[g] = dist / num_hits

                # round ri and remove hits with ri == 0.00
                keys_for_deletion = set()
                for p in prediction:
                    ri = round(prediction[p], 2)
                    if ri == 0.00:
                        keys_for_deletion.add(p)
                    else:
                        prediction[p] = ri

                for j in keys_for_deletion:
                    del prediction[j]

                # reduce prediction to leaf terms
                parent_terms = []
                for p in prediction.keys():
                    parents = self.gen_ontology.get_parent_terms(p)
                    parent_terms += parents

                # exclude terms that are parent terms, i.e. there are more specific terms also part of this prediction
                keys_for_deletion = set()
                for p in prediction.keys():
                    if p in parent_terms:
                        keys_for_deletion.add(p)

                for k in keys_for_deletion:
                    del prediction[k]

        return prediction

    @staticmethod
    def write_predictions(predictions, out_file):
        """Write prediictions.

        :param predictions: predictions to write
        :param out_file: output file
        :return:
        """
        with open(out_file, 'w') as out:
            for p in predictions.keys():
                prediction = predictions[p]
                for pred in prediction.keys():
                    ri = prediction[pred]
                    out.write('{}\t{}\t'.format(p, pred))
                    out.write('{:0.2f}\n'.format(float(ri)))
