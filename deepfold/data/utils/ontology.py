import math
from collections import Counter, deque


class Ontology(object):
    """
    [Term]
    id: GO:0000003
    name: reproduction
    namespace: biological_process
    alt_id: GO:0019952
    alt_id: GO:0050876
    def: "The production of new individuals that contain some portion of genetic material \
        inherited from one or more parent organisms." [GOC:go_curators, GOC:isa_complete,\
        GOC:jl, ISBN:0198506732]
    subset: goslim_agr
    subset: goslim_chembl
    subset: goslim_flybase_ribbon
    subset: goslim_pir
    subset: goslim_plant
    synonym: "reproductive physiological process" EXACT []
    xref: Wikipedia:Reproduction
    is_a: GO:0008150 ! biological_process
    disjoint_from: GO:0044848 ! biological phase
    """
    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ontology = self.load_obo(filename, with_rels=with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ontology

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ontology[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_anchestors(self, term_id):
        if term_id not in self.ontology:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ontology[t_id]['is_a']:
                    if parent_id in self.ontology:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ontology:
            return set()
        term_set = set()
        for parent_id in self.ontology[term_id]['is_a']:
            if parent_id in self.ontology:
                term_set.add(parent_id)
        return term_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, goobj in self.ontology.items():
            if goobj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ontology[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ontology:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ontology[t_id]['children']:
                    q.append(ch_id)
        return term_set

    def load_obo(self, filename, with_rels=False):
        ontlogy = dict()
        goobj = None
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if goobj is not None:
                        ontlogy[goobj['id']] = goobj

                    goobj = dict()
                    goobj['is_a'] = list()
                    goobj['part_of'] = list()
                    goobj['regulates'] = list()
                    goobj['alt_ids'] = list()
                    goobj['is_obsolete'] = False
                    continue

                elif line == '[Typedef]':
                    if goobj is not None:
                        ontlogy[goobj['id']] = goobj
                    goobj = None

                else:
                    if goobj is None:
                        continue

                    subline = line.split(': ')
                    if subline[0] == 'id':
                        goobj['id'] = subline[1]
                    elif subline[0] == 'alt_id':
                        goobj['alt_ids'].append(subline[1])
                    elif subline[0] == 'namespace':
                        goobj['namespace'] = subline[1]
                    elif subline[0] == 'is_a':
                        goobj['is_a'].append(subline[1].split(' ! ')[0])
                    elif with_rels and subline[0] == 'relationship':
                        it = subline[1].split()
                        # add all types of relationships
                        goobj['is_a'].append(it[1])
                    elif subline[0] == 'name':
                        goobj['name'] = subline[1]
                    elif subline[0] == 'is_obsolete' and subline[1] == 'true':
                        goobj['is_obsolete'] = True
            if goobj is not None:
                ontlogy[goobj['id']] = goobj
            for term_id in list(ontlogy.keys()):
                for alt_id in ontlogy[term_id]['alt_ids']:
                    ontlogy[alt_id] = ontlogy[term_id]
                if ontlogy[term_id]['is_obsolete']:
                    del ontlogy[term_id]

            for term_id, val in ontlogy.items():
                if 'children' not in val:
                    val['children'] = set()
                for p_id in val['is_a']:
                    if p_id in ontlogy:
                        if 'children' not in ontlogy[p_id]:
                            ontlogy[p_id]['children'] = set()
                        ontlogy[p_id]['children'].add(term_id)
        return ontlogy


if __name__ == '__main__':
    gofile = '/Users/robin/xbiome/datasets/xbiome/go.obo'
    go = Ontology(filename=gofile)
    ontology = go.ontology
    print(ontology)