from catalyst import metrics
import torch
import torch.nn as nn
from typing import *
from const import Config
from torchmetrics.classification import BinaryPrecision, BinaryRecall


class CustomAccuracyMetric(metrics.ICallbackBatchMetric, metrics.AdditiveMetric):
    """Calculates accuracy metric that is used in Catalyst runner"""

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        self.sigmoid = nn.Sigmoid()
        value = ((self.sigmoid(scores) > 0.5) == targets).float().mean().item()
        value = super().update(value, len(targets))
        return value

    def update_key_value(self, scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        value = self.update(scores, targets)
        return {"accuracy": value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {"accuracy": mean, "accuracy/std": std}


class CustomPrecisionMetric(metrics.ICallbackBatchMetric, metrics.AdditiveMetric):
    """Calculates precision metric that is used in Catalyst runner"""

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        self.sigmoid = nn.Sigmoid()
        self.metric = BinaryPrecision(threshold=0.6).to(Config.DEVICE)
        value = self.metric(self.sigmoid(scores), targets).cpu()
        value = super().update(value, len(targets))
        return value

    def update_key_value(self, scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        value = self.update(scores, targets)
        return {"precision": value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {"precision": mean}


class CustomRecallMetric(metrics.ICallbackBatchMetric, metrics.AdditiveMetric):
    """Calculates recall metric that is used in Catalyst runner"""

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        self.sigmoid = nn.Sigmoid()
        self.metric = BinaryRecall(threshold=0.3).to(Config.DEVICE)
        value = self.metric(self.sigmoid(scores), targets).cpu()
        value = super().update(value, len(targets))
        return value

    def update_key_value(self, scores: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        value = self.update(scores, targets)
        return {"recall": value}

    def compute_key_value(self) -> Dict[str, float]:
        mean, std = super().compute()
        return {"recall": mean}