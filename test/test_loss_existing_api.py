"""Compatibility and correctness checks for LightOp loss exports."""

import pytest
import torch

import lightop
import lightop.loss


EXISTING_LOSS_APIS = ["CTCLoss", "RNNTLoss"]


def _rnnt_reference(logits, targets, input_lengths, target_lengths, blank=0):
    """Small differentiable RNNT reference using the forward algorithm."""
    log_probs = logits.log_softmax(-1)
    losses = []
    for batch_index in range(logits.shape[0]):
        time = int(input_lengths[batch_index])
        target_size = int(target_lengths[batch_index])
        alpha = [[None] * (target_size + 1) for _ in range(time)]
        alpha[0][0] = log_probs.new_zeros(())
        for t in range(time):
            for u in range(target_size + 1):
                if t == 0 and u == 0:
                    continue
                terms = []
                if t > 0:
                    terms.append(
                        alpha[t - 1][u]
                        + log_probs[batch_index, t - 1, u, blank]
                    )
                if u > 0:
                    label = targets[batch_index, u - 1].long()
                    terms.append(
                        alpha[t][u - 1]
                        + log_probs[batch_index, t, u - 1, label]
                    )
                alpha[t][u] = torch.logsumexp(torch.stack(terms), dim=0)
        log_likelihood = (
            alpha[time - 1][target_size]
            + log_probs[batch_index, time - 1, target_size, blank]
        )
        losses.append(-log_likelihood)
    return torch.stack(losses).mean()


@pytest.mark.parametrize("name", EXISTING_LOSS_APIS)
def test_existing_loss_api_is_direct_reexport(name):
    assert getattr(lightop.loss, name) is getattr(lightop, name)


def test_loss_public_api_inventory():
    assert set(lightop.loss.__all__) == set(EXISTING_LOSS_APIS)


def test_ctc_loss_matches_torch_forward_and_backward():
    torch.manual_seed(13)
    time, batch, classes, target_size = 12, 3, 9, 4
    logits = torch.randn(time, batch, classes, device="cuda")
    targets = torch.randint(
        1, classes, (batch, target_size), dtype=torch.int32, device="cuda"
    )
    input_lengths = torch.full(
        (batch,), time, dtype=torch.int32, device="cuda"
    )
    target_lengths = torch.full(
        (batch,), target_size, dtype=torch.int32, device="cuda"
    )

    actual_input = logits.transpose(0, 1).contiguous().requires_grad_()
    actual = lightop.loss.CTCLoss(
        blank=0, reduction="mean", logits_time_major=False
    )(actual_input, targets, input_lengths, target_lengths)
    actual.backward()

    expected_input = logits.detach().clone().requires_grad_()
    expected = torch.nn.CTCLoss(blank=0, reduction="mean")(
        expected_input.log_softmax(-1),
        targets,
        input_lengths,
        target_lengths,
    )
    expected.backward()

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        actual_input.grad.transpose(0, 1),
        expected_input.grad,
        rtol=1e-5,
        atol=1e-5,
    )


def test_rnnt_loss_matches_reference_forward_and_backward():
    torch.manual_seed(14)
    batch, time, target_size, classes = 2, 6, 3, 8
    logits = torch.randn(
        batch, time, target_size + 1, classes, device="cuda"
    )
    targets = torch.randint(
        1, classes, (batch, target_size), dtype=torch.int32, device="cuda"
    )
    input_lengths = torch.full(
        (batch,), time, dtype=torch.int32, device="cuda"
    )
    target_lengths = torch.full(
        (batch,), target_size, dtype=torch.int32, device="cuda"
    )

    actual_input = logits.detach().clone().requires_grad_()
    actual = lightop.loss.RNNTLoss(blank=0, reduction="mean")(
        actual_input, targets, input_lengths, target_lengths
    )
    actual.backward()

    expected_input = logits.detach().clone().requires_grad_()
    expected = _rnnt_reference(
        expected_input, targets, input_lengths, target_lengths, blank=0
    )
    expected.backward()

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-5
    )
