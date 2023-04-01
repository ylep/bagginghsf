import numpy as np
import torch
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric

y = torch.randint(0, 4, (1, 16, 16, 16))
y_hat = y * .8 - 1

metrics = [
    metric.DiceCoefficient(),
    metric.VolumeSimilarity(),
    metric.MahalanobisDistance()
]
labels = {
    0: 'background',
    1: 'edema',
    2: 'non-enhancing tumor',
    3: 'enhancing tumor'
}
evaluator = eval_.SegmentationEvaluator(metrics, labels)
evaluator.evaluate(y_hat, y, "sub00")

{
    r.label.title() + " " + r.metric.title(): r.value
    for r in evaluator.results
    if r.label != 'background'
}

np.mean([
    r.value
    for r in evaluator.results
    if (r.label != 'background' and r.metric == 'DICE')
])
