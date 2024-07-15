import torch
import torch.nn as nn

class TopKPooling_1(nn.Module):
    def __init__(
            self,
            k: object,
            dim: object
    ) -> object:
        super().__init__()
        self.k = k
        self.dim = dim

    def maxk_pool(self, x, reduce_dim, k):
        """ Performs max-k pooling along a given `reduce_dim` """
        index = x.topk(k, dim=reduce_dim)[1]
        return x.gather(reduce_dim, index)

    def forward(self, x, attention_mask=None):
        k = self.k
        if attention_mask is not None:
            x[torch.where(attention_mask == 0)] = -10000
            min_length = min(attention_mask.sum(1))
            if min_length < k:
                k = min_length
        maxk_selected_x = self.maxk_pool(x, self.dim, k)
        return maxk_selected_x.mean(self.dim)
