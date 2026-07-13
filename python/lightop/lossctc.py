import torch
#from lightop import op
from . import op


class CTCLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural Networks
    """

    def __init__(self, blank: int = 0, clamp: int = -1, reduction: str = 'mean',
                 logits_time_major: bool = True, check_lengths: bool = True):
        """
        :param blank:  (int, optional) – blank label (Default: 0)
        :param clamp: (float, optional) – clamp for gradients (Default: -1)
        :param reduction: (string, optional) – Specifies the reduction to apply to the output:
                        "none" | "mean" | "sum". (Default: "mean")
        :param logits_time_major: true - logits shape (T,N,C); false -- logits shape (N,T,C);
        :param check_lengths: true - check T==max(logit_lengths), S==max(target_lengths)
        """
        super(CTCLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.compute_sum = False
        self.compute_mean = False
        if reduction == 'sum':
            self.compute_sum = True
        if reduction == 'mean':
            self.compute_mean = True

        self.blank = blank
        self.clamp = clamp
        self.logits_time_major = logits_time_major
        self.check_lengths = check_lengths

    def forward(self, logits, targets, logit_lengths, target_lengths):
        """Forward operation

        Arguments:
            logits (tensor): input tensor to the loss function, shape of (T, N, C) or (N, T, C).
            targets (tensor): labels for the input data, shape of (N, S) or sum(target_lengths)
            logit_lengths (tensor): lengths of the inputs in the time dimension for each batch.
            target_lengths (tensor): lengths of the labels for each batch.
        """
        return op.accelerate_ctc_loss(logits, targets, logit_lengths, target_lengths,
                                      self.blank, self.clamp, self.logits_time_major, self.compute_sum, self.compute_mean, self.check_lengths)
