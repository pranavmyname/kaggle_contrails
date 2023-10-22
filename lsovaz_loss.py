import torch
import torch.nn as nn

weights = [ 1.428, 40.097]
class Lovasz(nn.Module):
    """Lovasz Loss. Cf: https://arxiv.org/abs/1705.08790 """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()
        # assert C >= 2, "Classification imply at least two Classes"

        loss = 0.0
        non_empty_C = 0

        for c in range(C):

            inputs_class = inputs[:, c]
            masks = (targets==c).float()

            for mask, input_class in zip(masks.view(N, -1), inputs_class.view(N, -1)):

                if mask.sum() == 0 and (input_class > 0.25).sum() == 0:
                    continue

                distance = (mask - input_class).abs()
                distance_sorted, indices = torch.sort(distance, 0, descending=True)
                mask_sorted = mask[indices.data]

                inter = mask_sorted.sum() - mask_sorted.cumsum(0)
                union = mask_sorted.sum() + (1.0 - mask_sorted).cumsum(0)
                iou = 1.0 - inter / union

                p = len(mask_sorted)
                iou[1:p] = iou[1:p] - iou[0:-1]

                loss += torch.dot(distance_sorted, iou)*weights[c]
                non_empty_C += 1

        return loss / N / non_empty_C