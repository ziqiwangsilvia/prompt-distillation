
import torch
from collections import defaultdict
from typing import List, Dict, Tuple

class RunningAverageTensor:
    """Tracks running averages of tensor statistics by index."""
    def __init__(self, total: torch.Tensor, count: torch.Tensor):
        assert total.shape == count.shape
        self.total = total
        self.count = count

    def add(self, dim: int, index: torch.Tensor, values: torch.Tensor) -> None:
        """Add values at specified indices along given dimension."""
        self.total.index_add_(dim, index, values.to(self.total.device, dtype=self.total.dtype))
        self.count.index_add_(dim, index, torch.ones_like(values).to(dtype=self.count.dtype))

    def get_average(self) -> torch.Tensor:
        """Return element-wise averages (0 where count is zero)."""
        total = self.total
        count = self.count
        return torch.where(count != 0, total / count, torch.zeros_like(total))

    def get_total_average(self) -> float:
        """Return average across all elements."""
        total = self.total.sum()
        count = self.count.sum()
        return 0.0 if count == 0 else float(total / count)

    def __repr__(self) -> str:
        return f"RunningAverageTensor(total={self.total}, count={self.count})"

    def to(self, device) -> None:
        """Move tensors to the specified device."""
        self.total = self.total.to(device)
        self.count = self.count.to(device)

class Aggregator:
    """
    Aggregates metrics for every sample in the validation set,
    supports group-wise and total statistics.
    """
    def __init__(self, group_names: List[str], device):
        self.group_names = group_names
        def ra_factory():
            total = torch.zeros(len(group_names), device=device)
            count = torch.zeros(len(group_names), device=device)
            return RunningAverageTensor(total, count)
        self._ra = defaultdict(ra_factory)

    def to(self, device) -> None:
        """Move all statistics to the specified device."""
        for v in self._ra.values():
            v.to(device)

    def add_batch(
        self,
        batch_group_ixs: torch.Tensor,
        batch_metrics: Dict[str, torch.Tensor],
        accelerator
    ) -> None:
        """
        Add a batch of group indices and associated metrics.
        Uses accelerator.gather_for_metrics to sync across devices.
        """
        metric_names = list(batch_metrics.keys())
        batch_metrics["group_ixs"] = batch_group_ixs
        gathered_metrics = accelerator.gather_for_metrics(batch_metrics)
        for metric_name in metric_names:
            values = gathered_metrics[metric_name]
            group_ixs = gathered_metrics["group_ixs"]
            self._ra[metric_name].add(0, group_ixs.flatten(), values.flatten())

    def key_to_string(self, key: Tuple[str, int]) -> str:
        """Convert (metric_name, group_id) to a readable string."""
        metric_name, group_id = key
        group_name = self.group_names[int(group_id)]
        return "/".join((metric_name, group_name))

    def get_average(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns (metrics_total, metrics_by_group) where:
        - metrics_total: metric_name -> float
        - metrics_by_group: "metric/group_name" -> float
        """
        metrics_by_group = {}
        metrics_total = {}
        for metric_name, ra in self._ra.items():
            av = ra.get_average()
            metrics_by_group.update({
                self.key_to_string((metric_name, i)): float(av[i])
                for i in range(len(self.group_names))
                if ra.count[i] > 0
            })
            metrics_total[metric_name] = ra.get_total_average()
        return metrics_total, metrics_by_group
