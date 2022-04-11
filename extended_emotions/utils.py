import torch 
from tabulate import tabulate

class ClassificationMetrics:
    """Accumulate per-class confusion matrices for a classification task."""
    metrics = ('accuracy', 'recall', 'precision', 'f1_score', 'iou')

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = self.fn = self.fp = self.tn = 0

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        return self.tp + self.fn

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.tp + self.fn
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        # return self.count.sum()
        return (self.tp + self.fn).sum()

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        pred, true = pred.flatten(), true.flatten()
        classes = torch.arange(0, self.num_classes, device=true.device)
        valid = (0 <= true) & (true < self.num_classes)
        pred_pos = classes.view(-1, 1) == pred[valid].view(1, -1)
        positive = classes.view(-1, 1) == true[valid].view(1, -1)
        pred_neg, negative = ~pred_pos, ~positive
        self.tp += (pred_pos & positive).sum(dim=1)
        self.fp += (pred_pos & negative).sum(dim=1)
        self.fn += (pred_neg & positive).sum(dim=1)
        self.tn += (pred_neg & negative).sum(dim=1)

    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = self.fn = self.fp = self.tn = 0

    @property
    def accuracy(self):
        """Get the per-class accuracy."""
        # we avoid dividing by zero using: max(denominator, 1)
        return (self.tp + self.tn) / self.total.clamp(min=1)

    @property
    def recall(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fn).clamp(min=1)

    @property
    def precision(self):
        """Get the per-class precision."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def f1_score(self):  # a.k.a. Sorensen–Dice Coefficient
        """Get the per-class F1 score."""
        # we avoid dividing by zero using: max(denominator, 1)
        tp2 = 2 * self.tp
        return tp2 / (tp2 + self.fp + self.fn).clamp(min=1)

    @property
    def iou(self):
        """Get the per-class intersection over union."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def weighted(self, scores):
        """Compute the weighted sum of per-class metrics."""
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
        """Quick hack to add mean and weighted properties."""
        if name.startswith('mean_') or name.startswith('weighted_'):
            metric = getattr(self, '_'.join(name.split('_')[1:]))
            if name.startswith('mean_'):
                return metric.mean()
            else:
                return self.weighted(metric)
        raise AttributeError(name)

    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = torch.stack([getattr(self, m) for m in self.metrics])

        perc = lambda x: f'{float(x) * 100:.2f}%'.rjust(8)
        out = '   Class  ' + ' '.join(map(lambda x: x.rjust(7), self.metrics))

        out += '\n' + '-' * 53
        for i, values in enumerate(metrics.t()):
            out += '\n' + str(i).rjust(8) + ' '
            out += ' '.join(map(lambda x: perc(x.mean()), values))
        out += '\n' + '-' * 53

        out += '\n    Mean '
        out += ' '.join(map(lambda x: perc(x.mean()), metrics))

        out += '\nWeighted '
        out += ' '.join(map(lambda x: perc(self.weighted(x)), metrics))
        return out


class ConfusionMatrix:
    """Accumulate a confusion matrix for a classification task."""

    def __init__(self, num_classes, class_labels=None):
        self.value = 0
        self.num_classes = num_classes
        self.class_labels = class_labels if class_labels else range(self.num_classes)
        

    @torch.no_grad()
    def update(self, pred, true):  # doesn't allow for "ignore_index"
        """Update the confusion matrix with the given predictions."""
        unique_mapping = true.flatten() * self.num_classes + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.num_classes**2)
        self.value += bins.view(self.num_classes, self.num_classes)

    def reset(self):
        """Reset all accumulated values."""
        self.value = 0

    @property
    def tp(self):
        """Get the true positive samples per-class."""
        return self.value.diag()

    @property
    def fn(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=1) - self.value.diag()

    @property
    def fp(self):
        """Get the false positive samples per-class."""
        return self.value.sum(dim=0) - self.value.diag()

    @property
    def tn(self):
        """Get the true negative samples per-class."""
        # return self.total - (self.tp + self.fn + self.fp)
        # this is the same as the above but ~slightly~ more efficient
        tp = self.value.diag()
        actual = self.value.sum(dim=1)  # tp + fn
        predicted = self.value.sum(dim=0)  # tp + fp
        # rest = actual + predicted - tp  # tp + fn + fp
        # return actual.sum() - rest
        return actual.sum() + tp - (actual + predicted)

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        # return self.tp + self.fn
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)
    
    @property
    def total(self):
        """Get the total number of samples."""
        return self.value.sum()
    
    @property
    def accuracy(self):
        """Compute the per-class accuracy."""
        return (self.tp + self.tn) / self.total.clamp(min=1)

    @property
    def recall(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fn).clamp(min=1)
    
    def weighted(self, scores):
        """Compute the weighted sum of per-class metrics."""
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
        """Quick hack to add mean and weighted properties."""
        if name.startswith('mean_') or name.startswith('weighted_'):
            metric = getattr(self, '_'.join(name.split('_')[1:]))
            if name.startswith('mean_'):
                return metric.mean()
            else:
                return self.weighted(metric)
        raise AttributeError(name)
    
    def __repr__(self):
        """A tabular representation of the metrics."""
        out = tabulate(self.value.cpu().numpy(), headers=self.class_labels,
                       showindex=self.class_labels, tablefmt='fancy_grid')
        return out