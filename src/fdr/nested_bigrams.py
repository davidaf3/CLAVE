import ast
from collections import Counter


def code_bigram(node):
    result = []
    if node.__class__ == ast.Module:
        for child in ast.iter_child_nodes(node):
            bigram_collector = code_bigram(child)
            result = result + bigram_collector
    else:
        for child in ast.iter_child_nodes(node):
            bigram = str((ast.dump(node), ast.dump(child)))
            result.append(bigram)
            bigram_collector = code_bigram(child)
            if not bigram_collector == []:
                result = result + bigram_collector
    return result


def get_bigrams(program):
    tree = ast.parse(program)
    return dict(Counter(code_bigram(tree)))


def get_feature_vector(program, vocab):
    features = []
    bigrams = get_bigrams(program)
    for bigram in vocab:
        freq = bigrams[bigram] if bigram in bigrams else 0
        features.append(freq)

    return features
