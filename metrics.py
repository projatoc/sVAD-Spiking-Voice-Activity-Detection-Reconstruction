"""Frame-level HTER, with speech as the positive class (label=1)."""
import torch


def confusion_counts(logits, labels):
    predictions = logits.argmax(-1).reshape(-1)
    targets = labels.reshape(-1)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    tn = int(((predictions == 0) & (targets == 0)).sum())
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}


def rates(c):
    speech = c['tp'] + c['fn']
    nonspeech = c['fp'] + c['tn']
    mr = c['fn']/speech if speech else float('nan')
    far = c['fp']/nonspeech if nonspeech else float('nan')
    return {'mr': mr, 'far': far,
            'hter': (mr+far)/2 if speech and nonspeech else float('nan'),
            'speech_frames': speech, 'nonspeech_frames': nonspeech}
