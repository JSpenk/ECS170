'''
Concrete Evaluate class: multi-class Accuracy, Precision, Recall and F1
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class Evaluate_Metrics(evaluate):
    data = None

    @staticmethod
    def _to_numpy(v):
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        return np.asarray(v)

    def evaluate(self):
        '''
        Returns a dict of scores. This is a multi-class task, so the binary versions of
        precision/recall/F1 cannot be used; we report the macro, weighted and micro averages.
        '''
        print('evaluating performance...')
        true_y = self._to_numpy(self.data['true_y'])
        pred_y = self._to_numpy(self.data['pred_y'])
        scores = {'accuracy': accuracy_score(true_y, pred_y)}
        for avg in ['macro', 'weighted', 'micro']:
            scores['precision_' + avg] = precision_score(true_y, pred_y, average=avg, zero_division=0)
            scores['recall_' + avg] = recall_score(true_y, pred_y, average=avg, zero_division=0)
            scores['f1_' + avg] = f1_score(true_y, pred_y, average=avg, zero_division=0)
        return scores
