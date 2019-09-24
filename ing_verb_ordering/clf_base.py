from ing_verb_ordering.constants import OFFSET

import operator

# use this to find the highest-scoring label
argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    d = {}
    for w, c in base_features.items():
        d[(label, w)] = c
    d[(label, OFFSET)] = 1
    return d
    

def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}
    for label in labels:
        dot_product = 0
        val = 0
        for b in base_features.keys():
            val = base_features[b] * weights[(label, b)]
            dot_product += val
        if (label, OFFSET) in weights.keys():
            off = weights[(label, OFFSET)]
            dot_product += weights[(label, OFFSET)]  # *base_features[(label, "**OFFSET**")]
        scores[label] = dot_product
    return argmax(scores), scores

    
