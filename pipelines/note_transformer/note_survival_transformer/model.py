"""Compact causal note encoder with an explicit landmark-conditioning prefix."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import Config


class NoteSurvivalTransformer(nn.Module):
    """Return log-relative-risk [B], never probabilities.

    attention_mask: bool [B,L], True means a valid note (opposite of PyTorch's
    key-padding mask). Valid histories must be chronologically ordered. Forward
    also filters future notes and caps history, protecting direct API callers.
    No positional embeddings are used; order, causal attention and continuous
    time features supply longitudinal information.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config.validate(require_dimensions=True)
        d = config.d_model
        self.note_projection = nn.Linear(config.input_dim, d)
        self.time_projection = nn.Linear(2, 8)
        self.gap_projection = nn.Linear(1, 8)
        self.input_projection = nn.Linear(d + 16, d)
        self.landmark_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.landmark_token, std=0.02)
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(d, config.n_heads, config.dim_ff,
                                           config.dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.n_layers,
                                             enable_nested_tensor=False)
        if config.static_dim:
            self.static_encoder = nn.Sequential(nn.Linear(config.static_dim, d), nn.ReLU(),
                                                nn.Dropout(config.dropout), nn.Linear(d, d))
            self.static_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(),
                                               nn.Dropout(config.dropout))
        else:
            self.static_encoder = None
        self.risk_head = nn.Linear(d, 1)

    def forward(self, x_notes, days, landmark, attention_mask, static_covariates=None):
        cfg = self.config
        if x_notes.ndim != 3 or x_notes.shape[2] != cfg.input_dim:
            raise ValueError("x_notes must be [B,L,input_dim]")
        b, length, _ = x_notes.shape
        if b == 0 or days.shape != (b, length) or attention_mask.shape != (b, length):
            raise ValueError("days/mask must match a nonempty [B,L] batch")
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be bool, True = valid note")
        if not x_notes.is_floating_point() or not days.is_floating_point():
            raise ValueError("Note vectors and timestamps must be floating point")
        lm = torch.as_tensor(landmark, dtype=days.dtype, device=days.device)
        if lm.ndim == 0:
            lm = lm.expand(b)
        if lm.shape != (b,) or not torch.isfinite(lm).all():
            raise ValueError("landmark must be a finite scalar or [B]")
        if not torch.isfinite(days[attention_mask]).all():
            raise ValueError("Valid note timestamps must be finite")
        valid = attention_mask & (days <= lm[:, None])
        vectors, timestamps, counts = [], [], []
        for row in range(b):
            positions = torch.nonzero(valid[row], as_tuple=False).flatten()
            selected_days = days[row, positions]
            if selected_days.numel() > 1 and (selected_days[1:] < selected_days[:-1]).any():
                raise ValueError("Valid pre-landmark notes must be chronologically sorted")
            positions = positions[-cfg.max_notes:]
            selected = x_notes[row, positions]
            if not torch.isfinite(selected).all():
                raise ValueError("Pre-landmark note embeddings must be finite")
            vectors.append(selected)
            timestamps.append(days[row, positions])
            counts.append(positions.numel())
        x = pad_sequence(vectors, batch_first=True)
        note_days = pad_sequence(timestamps, batch_first=True)
        counts = torch.tensor(counts, device=x.device, dtype=torch.long)
        n = x.shape[1]
        mask = torch.arange(n, device=x.device)[None, :] < counts[:, None]
        lm_feature = torch.tanh(lm / cfg.landmark_scale_days)
        relative = torch.tanh((note_days - lm[:, None]) / cfg.time_scale_days)
        abs_features = torch.stack([relative, lm_feature[:, None].expand(-1, n)], dim=-1)
        gaps = torch.zeros_like(note_days)
        if n > 1:
            gaps[:, 1:] = (note_days[:, 1:] - note_days[:, :-1]).clamp(min=0, max=cfg.gap_cap_days)
        gaps = torch.log1p(gaps) / torch.log1p(gaps.new_tensor(cfg.gap_cap_days))
        gaps = gaps.masked_fill(~mask, 0.)
        notes = self.note_projection(x)
        prefix = self.landmark_token.expand(b, -1, -1)
        prefix_time = torch.stack([lm_feature, lm_feature], dim=-1)[:, None, :]
        content = torch.cat([prefix, notes], dim=1)
        time = torch.cat([prefix_time, abs_features], dim=1).to(dtype=content.dtype)
        gap = torch.cat([gaps.new_zeros(b, 1), gaps], dim=1).unsqueeze(-1).to(dtype=content.dtype)
        encoded_input = self.input_projection(torch.cat([
            content, self.time_projection(time), self.gap_projection(gap)], dim=-1))
        encoded_input = self.input_dropout(self.input_norm(encoded_input))
        valid_tokens = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        causal = torch.triu(torch.ones(n + 1, n + 1, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(encoded_input, mask=causal, src_key_padding_mask=~valid_tokens)
        pooled = h[torch.arange(b, device=x.device), counts]  # prefix index 0 if no notes
        if self.static_encoder is not None:
            if static_covariates is None or static_covariates.shape != (b, cfg.static_dim):
                raise ValueError("static_covariates must be [B,static_dim]")
            if not torch.isfinite(static_covariates).all():
                raise ValueError("static_covariates must be finite after imputation")
            pooled = self.static_fusion(torch.cat([pooled, self.static_encoder(static_covariates)], dim=-1))
        elif static_covariates is not None:
            raise ValueError("This model was configured without static covariates")
        return self.risk_head(pooled).squeeze(-1)
