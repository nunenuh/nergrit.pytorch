import itertools
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from .conlleval import conll_evaluation

def ner_metrics_fn(list_hyp, list_label):
    metrics = {}
    acc, pre, rec, f1, tm_pre, tm_rec, tm_f1 = conll_evaluation(list_hyp, list_label)
    metrics["ACC"] = acc
    metrics["F1"] = tm_f1
    metrics["REC"] = tm_rec
    metrics["PRE"] = tm_pre
    return metrics