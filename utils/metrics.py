# -*- coding: utf-8 -*-
"""
Created on Wed June 7 00:36:17 2023
@title: eval metrics
"""
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


class EvalMetrics:
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr")
        ]

    def compute_scores(self, gts, res):
        msg = ''
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            msg += f'{str(method)}: {score}'
        return msg
