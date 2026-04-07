import numpy as np
from datasets import load_dataset
from typing import List, Optional
from core.utils import load_squadshifts


class ContextDistractorDataset:
    """Dataset for sampling random paragraphs as distractors."""
    def __init__(self, contexts: List[str]):
        self.contexts = contexts
        self.len = len(self.contexts)
        self.indices = list(range(self.len))
        self.reset()

    def __len__(self) -> int:
        return self.len

    def reset(self) -> None:
        """Shuffle indices and reset pointer."""
        self.ptr = 0
        np.random.shuffle(self.indices)

    def _sample(self, paragraphs: int = 2) -> List[str]:
        """Sample a given number of random paragraphs."""
        context = []
        for _ in range(paragraphs):
            context.append(self.contexts[self.indices[self.ptr]])
            self.ptr += 1
            if self.ptr == self.len:
                self.reset()
        return context

    def sample(self) -> List[str]:
        """Sample default number of random paragraphs."""
        return self._sample()


def build_distractor_dataset(
    dataset: str = None,
    n_distractor_dataset_items: int = 1000,
) -> ContextDistractorDataset:
    """Build a distractor dataset from SQuADShifts."""
    ds = load_squadshifts(dataset)
    prev_context = None
    contexts = []
    for i, item in enumerate(ds):
        if i >= n_distractor_dataset_items:
            break
        context = item['context'] 
        if context == prev_context:
            continue
        prev_context = context
        contexts.append(context)
    return ContextDistractorDataset(contexts)
