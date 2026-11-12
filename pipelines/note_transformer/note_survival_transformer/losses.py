"""Numerically stable sampled Cox and independent full Breslow reference."""

import torch


def sampled_cox_loss(log_risk, weights=None):
    """One case-first event group. Weights estimate a denominator, not an unbiased log loss."""
    if log_risk.ndim != 1 or log_risk.numel() == 0 or not torch.isfinite(log_risk).all():
        raise ValueError("log_risk must be a nonempty finite vector with the case first")
    if weights is None:
        weights = torch.ones_like(log_risk)
    weights = torch.as_tensor(weights, dtype=log_risk.dtype, device=log_risk.device)
    if weights.shape != log_risk.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Denominator weights must be positive and match log_risk")
    if weights[0].item() != 1.:
        raise ValueError("The case denominator weight must be one")
    return torch.logsumexp(log_risk + torch.log(weights), dim=0) - log_risk[0]


def full_breslow_loss(log_risk, time, event):
    """Full fixed-landmark cohort, normalized by events; inclusive ties, no Efron.

    Sorting/cumulative denominators give O(N log N) work. Every tied event uses
    the denominator at the END of its equal-time block in descending time order.
    Returns differentiable zero when no events are present; evaluation reports NaN.
    """
    if log_risk.ndim != 1 or time.shape != log_risk.shape or event.shape != log_risk.shape:
        raise ValueError("Scores, times and events must be aligned vectors")
    if not torch.isfinite(log_risk).all() or not torch.isfinite(time).all():
        raise ValueError("Scores and observed times must be finite")
    if not ((event == 0) | (event == 1)).all():
        raise ValueError("event must be binary")
    n_events = event.sum()
    if n_events.item() == 0:
        return log_risk.sum() * 0.
    order = torch.argsort(time, descending=True)
    times, scores, events = time[order], log_risk[order], event[order].to(log_risk.dtype)
    _, counts = torch.unique_consecutive(times, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    denominator = torch.logcumsumexp(scores, 0)[ends].repeat_interleave(counts)
    return ((denominator - scores) * events).sum() / n_events
