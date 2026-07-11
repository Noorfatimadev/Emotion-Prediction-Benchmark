from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class CollateFunctor:
    """
    Collate function used by DataLoader for variable-length sequences.

    Responsibilities:
    - pad sequences to a fixed max_length
    - truncate sequences that are too long
    - stack labels into shape [B, 1]

    Output:
        input_ids : LongTensor [B, L]
        labels    : FloatTensor [B, 1]

    where:
        B = batch size
        L = max_length
    """
    pad_idx: int      # token id used for padding
    max_length: int   # fixed sequence length used by the model

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        batch = list of (sequence, label) pairs from Dataset.

        Each sequence can have different length, so we:
          1) create a padded tensor filled with pad_idx
          2) copy each sequence into it
          3) truncate if sequence is longer than max_length
        """
        # Unpack batch into separate lists
        sequences, labels = zip(*batch)

        # Stack regression labels into [B, 1]
        labels = torch.stack(labels).view(-1, 1).float()

        B = len(sequences)        # batch size
        L = self.max_length       # target padded length

        # Initialize padded tensor:
        # everything starts as PAD tokens
        input_ids = torch.full((B, L), self.pad_idx, dtype=torch.long)

        # Copy each sequence into the padded tensor
        for i, seq in enumerate(sequences):
            seq = seq[:L]  # truncate long sequences
            input_ids[i, : seq.numel()] = seq

        return input_ids, labels