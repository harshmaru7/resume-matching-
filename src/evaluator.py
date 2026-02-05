import json
from typing import List, Dict, Optional
from dataclasses import dataclasses
import numpy as np

@dataclass
class EvaluationResult:
    spearman_correlation: float
    precision_at_k: Dict[int,float]
    recall_at_k: Dict[int,float]
    ndcg: float
    mean_absolute_error: float
    details: Optional[Dict] = None

class Evaluator:
    LABEL_MAPPING = {
        'good': 1.0, 'good match': 1.0,
        'partial': 0.5, 'partial match': 0.5,
        'poor':0.0, 'poor match':0.0,
    }

    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold

    def _normalize_labels(self, labels: List) -> List[float]:
        normalized = []
        for label in labels:
            if isinstance(label, (int,float)):
                normalized.append(float(label))
            elif isinstance(label, str):
                normalized.append(self.LABEL_MAPPING.get(label.lower(), 0.0))
            else:
                normalized.append(0.0)
        return normalized

    def spearman_correlation(self, predictions: List[float], labels: List[float]) -> float:
        n = len(predictions)
        if n == 0:
            return 0.0
        pred_ranks = self._get_ranks(predictions)
        label_ranks = self._get_ranks(labels)

        def get_ranks(values):
            sorted_idx = sorted(range(len(values)), key=lambda i:values[i],reverse=True)
            ranks = [0.0]*len(values)
            for rank,idx in enumerate(sorted_idx,1):
                ranks[idx] = float(rank)
            return ranks
        pred_ranks = get_ranks(predictions)
        label_ranks = get_ranks(labels)

        d_squared = sum((p - l) ** 2 for p, l in zip(pred_ranks, label_ranks))
        correlation = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
        return correlation

    def precision_at_k(self, predictions: List[float], labels: List[float], k: int) -> float:
        if k <= 0 or not predictions:
            return 0.0

        sorted_pairs = sorted(zip(predictions, labels), key=lambda x: x[0], reverse=True)
        top_k = sorted_pairs[:min(k, len(sorted_pairs))]

        relevant = sum(1 for _, label in top_k if label >= self.relevance_threshold)
        return relevant / len(top_k)

    def ndcg(self, predictions: List[float], labels: List[float], k: Optional[int] = None) -> float:
        if not predictions:
            return 0.0

        k = k or len(predictions)

        sorted_pairs = sorted(zip(predictions, labels), key=lambda x: x[0], reverse=True)[:k]

        dcg = sum(label / np.log2(i + 2) for i, (_, label) in enumerate(sorted_pairs))

        ideal_sorted = sorted(labels, reverse=True)[:k]
        idcg = sum(label / np.log2(i + 2) for i, label in enumerate(ideal_sorted))

        return dcg / idcg if idcg > 0 else 0.0

    def mean_absolute_error(self, predictions: List[float], labels: List[float]) -> float:
        if not predictions:
            return 0.0
        return sum(abs(p - l) for p, l in zip(predictions, labels)) / len(predictions)

    def evaluate(self, predictions: List[float], labels: List, k_values: List[int] = [1, 3, 5]) -> EvaluationResult:
        normalized = self._normalize_labels(labels)

        return EvaluationResult(
            spearman_correlation=self.spearman_correlation(predictions, normalized),
            precision_at_k={k: self.precision_at_k(predictions, normalized, k) for k in k_values},
            recall_at_k={k: self._recall_at_k(predictions, normalized, k) for k in k_values},
            ndcg=self.ndcg(predictions, normalized),
            mean_absolute_error=self.mean_absolute_error(predictions, normalized),
        )

    def _recall_at_k(self, predictions: List[float], labels: List[float], k: int) -> float:
        total_relevant = sum(1 for l in labels if l >= self.relevance_threshold)
        if total_relevant == 0:
            return 1.0

        sorted_pairs = sorted(zip(predictions, labels), key=lambda x: x[0], reverse=True)
        top_k = sorted_pairs[:min(k, len(sorted_pairs))]
        relevant_in_k = sum(1 for _, label in top_k if label >= self.relevance_threshold)

        return relevant_in_k / total_relevant
