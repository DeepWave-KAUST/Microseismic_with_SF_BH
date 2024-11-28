"""
Module to compute the matching cost and solve the corresponding Linear Sum Assignment Problem.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    TODO: matching of variable targets
    Stated in the original DETR:
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    
    # TODO: weight_cost_class & weight_cost_location - 1. initial values; 2. if the values could be larger than 1.
    def __init__(self, weight_cost_class: float = 1, weight_cost_location: float = 1):
        """
        Create the matcher.

        Params:
            weight_cost_class: This is the relative weight of the classification loss in the matching cost.
            weight_cost_location: This is the relative weight of the location loss in the matching cost.
        """
        super().__init__()
        self.weight_cost_class = weight_cost_class
        self.weight_cost_location = weight_cost_location
        assert weight_cost_class != 0 or weight_cost_location != 0, "Cost weights cant both be 0."

    @torch.no_grad()
    def forward(self, outputs, labels_cls, labels_loc):
        """
        Perform the matching.

        Params:
            outputs: This is a dict that contains entries:
                "pred_logits": tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_locations": tensor of dim [batch_size, num_queries, 3] with the predicted source locations (X, Y, Z)

            labels_cls: tensor of dim [batch_size, num_queries, 1] containing the class labels

            labels_loc: tensor of dim [batch_size, num_queries, 3] containning the target locations

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_events)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # predicted probability for each class (without no_event class), size of (batch_size, num_queries, num_classes)
        out_prob = outputs["pred_logits"].softmax(-1)[:, :, :-1]
        # predicted locations for each query, size of (batch_size, num_queries, 3) where 3 represents X, Y and Z locations
        out_loc = outputs["pred_locations"]

        # Cost matrix should be in size of (batch_size, num_queries, num_queries).

        # Compute the classification cost.
        # Contrary to the loss computation, we use the absolute difference between the targets and predictions, rather than the NLL.
        cost_class = abs(labels_cls.transpose(1, 2).repeat(1, num_queries, 1) - out_prob.repeat(1, 1, num_queries))

        # Compute the location cost.
        cost_location = torch.cdist(out_loc, labels_loc, p=2)

        # Form the final cost matrix.
        C = self.weight_cost_class * cost_class + self.weight_cost_location * cost_location
        C = C.cpu()   # linear_sum_assignment is cpu-based.

        # indices is a list of length batch_size
        # each item in it is a tuple, e.g., (array([0, 1, 2, 3, 4]), array([4, 3, 2, 0, 1]))
        indices = [linear_sum_assignment(C[i]) for i in range(batch_size)]
        # output is a list of length batch_size
        # each item in it is a tuple, e.g., (tensor([0, 1, 2, 3, 4]), tensor([4, 3, 2, 0, 1]))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(weight_cost_class: float = 1, weight_cost_location: float = 1):
    return HungarianMatcher(weight_cost_class=weight_cost_class, weight_cost_location=weight_cost_location)


def get_pred_permutation_idx(indices):
    """
    Permute predictions following indices.
    """
    batch_idx = torch.cat([torch.full_like(pred, i) for i, (pred, _) in enumerate(indices)])
    pred_idx = torch.cat([pred for (pred, _) in indices])
    return batch_idx, pred_idx


def get_tgt_permutation_idx(indices):
    """
    Permute targets following indices.
    """
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
