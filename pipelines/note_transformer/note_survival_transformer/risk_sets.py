"""Uniform sampled controls from the FULL training cohort at actual event time."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EventGroup:
    case_id: str
    member_ids: list  # case is always index 0
    landmark: float
    event_time: float
    weights: np.ndarray
    eligible_control_count: int


class RiskSetSampler:
    def __init__(self, outcomes, train_ids, landmarks, max_controls=32, use_ipw=False):
        self.ids = list(sorted(train_ids))
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Training IDs must be nonempty and unique")
        if max_controls < 1:
            raise ValueError("max_controls must be positive")
        frame = outcomes.loc[self.ids]
        self.times = frame["time"].to_numpy(dtype=np.float64)
        self.events = frame["event"].to_numpy(dtype=np.int64)
        self.landmarks = np.asarray(sorted(landmarks), dtype=np.float64)
        if not len(self.landmarks) or not np.isfinite(self.landmarks).all():
            raise ValueError("At least one finite landmark is required")
        if not np.isfinite(self.times).all() or not np.isin(self.events, [0, 1]).all():
            raise ValueError("Risk-set outcomes must be finite first-event times and binary events")
        self.max_controls = int(max_controls)
        self.use_ipw = bool(use_ipw)
        self.sorted_idx = np.argsort(self.times, kind="mergesort")
        self.sorted_times = self.times[self.sorted_idx]
        primary = np.flatnonzero(self.events == 1)
        self.event_indices = np.array([i for i in primary if (self.landmarks < self.times[i]).any()], dtype=int)
        self.excluded_no_landmark = int(len(primary) - len(self.event_indices))
        self.singleton_events = int(sum(len(self.risk_indices(self.times[i])) == 1 for i in self.event_indices))

    def risk_indices(self, event_time):
        start = np.searchsorted(self.sorted_times, event_time, side="left")
        return self.sorted_idx[start:]

    def sample(self, case_index, rng):
        if case_index not in self.event_indices:
            raise ValueError("Case is not a primary event with an eligible pre-event landmark")
        event_time = float(self.times[case_index])
        landmark = float(rng.choice(self.landmarks[self.landmarks < event_time]))
        controls = self.risk_indices(event_time)
        controls = controls[controls != case_index]
        n = len(controls)
        m = min(n, self.max_controls)
        chosen = rng.choice(controls, size=m, replace=False) if m else np.empty(0, dtype=int)
        control_weight = (n / m) if self.use_ipw and m else 1.
        return EventGroup(self.ids[case_index], [self.ids[case_index]] + [self.ids[j] for j in chosen],
                          landmark, event_time, np.array([1.] + [control_weight] * m, dtype=np.float32), n)

    def epoch_groups(self, seed):
        # Independent deterministic epoch stream: resumable without a worker-local RNG.
        rng = np.random.default_rng(seed)
        for case in rng.permutation(self.event_indices):
            yield self.sample(int(case), rng)

    def summary(self):
        return {"n_train_patients": len(self.ids), "n_primary_events": int(self.events.sum()),
                "n_eligible_event_groups": len(self.event_indices),
                "excluded_no_eligible_landmark": self.excluded_no_landmark,
                "singleton_event_groups": self.singleton_events}
