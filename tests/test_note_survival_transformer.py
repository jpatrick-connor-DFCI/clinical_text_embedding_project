"""Statistical and integration tests; run with stdlib unittest, no test dependency."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.note_transformer.note_survival_transformer.config import Config
from pipelines.note_transformer.note_survival_transformer.dataset import StaticTransform, load_data
from pipelines.note_transformer.note_survival_transformer.evaluate import evaluate_model, safe_c_index, weighted_validation_score
from pipelines.note_transformer.note_survival_transformer.hparam_search import run_search, sample_plan
from pipelines.note_transformer.note_survival_transformer.losses import sampled_cox_loss, full_breslow_loss
from pipelines.note_transformer.note_survival_transformer.model import NoteSurvivalTransformer
from pipelines.note_transformer.note_survival_transformer.predict import final_test_evaluation, inference, restore_model
from pipelines.note_transformer.note_survival_transformer.risk_sets import RiskSetSampler
from pipelines.note_transformer.note_survival_transformer.synthetic_example import generate_synthetic
from pipelines.note_transformer.note_survival_transformer.train import fit, load_checkpoint, seed_everything, train_epoch


def small_config(**overrides):
    values = dict(d_model=8, n_heads=2, n_layers=1, dim_ff=16, dropout=0., max_notes=4,
                  landmarks=(-2., .5, 2.), num_epochs=2, patience=2, lr=.01,
                  batch_size=4, risk_set_max_controls=8)
    values.update(overrides)
    return Config(**values)


class LossTests(unittest.TestCase):
    def test_worked_example_and_ipw(self):
        scores = torch.log(torch.tensor([4., 2., 1.], dtype=torch.float64))
        self.assertAlmostEqual(float(sampled_cox_loss(scores)), np.log(7 / 4), places=12)
        self.assertAlmostEqual(float(sampled_cox_loss(scores, [1, 3, 3])), np.log(13 / 4), places=12)
        with self.assertRaises(ValueError):
            sampled_cox_loss(scores, [2, 1, 1])

    def test_full_breslow_all_controls_with_ties(self):
        scores = torch.tensor([.3, -.5, .7, 1., -1.], dtype=torch.float64, requires_grad=True)
        times = torch.tensor([2., 2., 2., 4., 5.], dtype=torch.float64)
        events = torch.tensor([1, 1, 0, 1, 0])
        contributions = []
        for case in torch.where(events == 1)[0]:
            controls = [i for i in range(5) if times[i] >= times[case] and i != case]
            contributions.append(sampled_cox_loss(scores[[int(case)] + controls]))
        self.assertTrue(torch.allclose(torch.stack(contributions).mean(), full_breslow_loss(scores, times, events)))
        # Two tied failures share a three-person denominator (including tied censor).
        self.assertAlmostEqual(float(full_breslow_loss(torch.zeros(3), torch.tensor([1., 1., 1.]),
                                                      torch.tensor([1, 1, 0]))), np.log(3), places=6)

    def test_extreme_scores_and_shift_invariance(self):
        scores = torch.tensor([1000., -1000., 999.], requires_grad=True)
        loss = sampled_cox_loss(scores)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.allclose(loss, sampled_cox_loss(scores.detach() - 1000), atol=5e-5))
        zero = full_breslow_loss(scores, torch.ones(3), torch.zeros(3))
        self.assertEqual(float(zero), 0.)

    def test_c_index_sanity_and_undefined(self):
        self.assertEqual(safe_c_index([1, 2, 3], [3, 2, 1], [1, 1, 1])[0], 1.)
        self.assertEqual(safe_c_index([1, 2, 3], [1, 2, 3], [1, 1, 1])[0], 0.)
        self.assertEqual(safe_c_index([1, 2, 3], [0, 0, 0], [1, 1, 1])[0], .5)
        self.assertEqual(safe_c_index([1, 1], [1, 2], [1, 1])[1], "no_comparable_pairs")
        self.assertEqual(safe_c_index([1, 2], [1, 2], [0, 0])[1], "no_primary_events")
        self.assertTrue(np.isnan(safe_c_index([], [], [])[0]))
        self.assertEqual(weighted_validation_score([{"c_index": .6, "n_eligible": 10},
                                                    {"c_index": .8, "n_eligible": 30}]), .75)


class RiskSetTests(unittest.TestCase):
    def setUp(self):
        self.outcomes = pd.DataFrame({"patient_id": ["early", "case", "tie", "late", "death", "outside"],
                                      "time": [2., 4., 4., 9., 3., 20.],
                                      "event": [0, 1, 1, 0, 0, 1],
                                      "death_event": [0, 0, 0, 0, 1, 0]}).set_index("patient_id")
        self.ids = ["early", "case", "tie", "late", "death"]

    def test_global_actual_time_membership(self):
        sampler = RiskSetSampler(self.outcomes, self.ids, [0., 3.5], 99)
        case = sampler.ids.index("case")
        group = sampler.sample(case, np.random.default_rng(1))
        self.assertEqual(group.member_ids[0], "case")
        self.assertEqual(set(group.member_ids), {"case", "tie", "late"})
        self.assertNotIn("outside", group.member_ids)
        self.assertNotIn("early", group.member_ids)
        self.assertNotIn("death", group.member_ids)
        self.assertEqual(group.event_time, 4.)
        self.assertLess(group.landmark, group.event_time)
        self.assertTrue(all(p in sampler.ids for p in group.member_ids))

    def test_uniform_global_sampling_and_resampling(self):
        names = ["case"] + [f"control{i}" for i in range(8)]
        frame = pd.DataFrame({"time": [1.] + [2.] * 8, "event": [1] + [0] * 8}, index=names)
        sampler = RiskSetSampler(frame, names, [0.], 2, True)
        rng, counts, compositions = np.random.default_rng(4), dict.fromkeys(names[1:], 0), set()
        for _ in range(1000):
            group = sampler.sample(sampler.ids.index("case"), rng)
            self.assertEqual(len(set(group.member_ids)), 3)
            self.assertTrue(np.array_equal(group.weights, [1, 4, 4]))
            compositions.add(tuple(group.member_ids))
            for control in group.member_ids[1:]:
                counts[control] += 1
        self.assertTrue(all(190 < count < 310 for count in counts.values()))
        self.assertGreater(len(compositions), 20)
        # None of these controls has to coexist with the case in a patient minibatch.
        self.assertEqual(set(counts), set(names[1:]))

    def test_landmark_eligibility_fractional_and_exclusion(self):
        frame = pd.DataFrame({"time": [-3., .25, 4., 10.], "event": [1, 1, 1, 0]}, index=["a", "b", "c", "d"])
        sampler = RiskSetSampler(frame, list(frame.index), [-2.5, -.5, .25, 2.75], 2)
        self.assertEqual(sampler.excluded_no_landmark, 1)
        rng = np.random.default_rng(12)
        observed = set()
        for _ in range(500):
            group = sampler.sample(sampler.ids.index("c"), rng)
            self.assertLess(group.landmark, group.event_time)
            observed.add(group.landmark)
        self.assertEqual(observed, {-2.5, -.5, .25, 2.75})
        for _ in range(20):
            self.assertLess(sampler.sample(sampler.ids.index("b"), rng).landmark, .25)
        with self.assertRaises(ValueError):
            sampler.sample(sampler.ids.index("a"), rng)

    def test_epoch_sampling_is_reproducible_and_changes(self):
        frame = pd.DataFrame({"time": [5.] * 12, "event": [1] * 12}, index=[f"p{i}" for i in range(12)])
        sampler = RiskSetSampler(frame, list(frame.index), [-1., .25, 3.5], 3)
        def signature(seed):
            return [(g.case_id, tuple(g.member_ids), g.landmark) for g in sampler.epoch_groups(seed)]
        self.assertEqual(signature(42), signature(42))
        self.assertNotEqual(signature(42), signature(43))


class ModelTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)
        self.config = small_config(input_dim=3, max_notes=2)
        self.model = NoteSurvivalTransformer(self.config).eval()

    def test_empty_variable_length_and_padding_invariance(self):
        x = torch.randn(2, 3, 3)
        days = torch.tensor([[-2., -1., 8.], [0., 0., 0.]])
        mask = torch.tensor([[True, True, False], [False, False, False]])
        with torch.no_grad():
            score = self.model(x, days, 0., mask)
            padded = self.model(torch.cat([x, torch.full((2, 5, 3), float("nan"))], dim=1),
                                torch.cat([days, torch.full((2, 5), float("nan"))], dim=1), 0.,
                                torch.cat([mask, torch.zeros(2, 5, dtype=torch.bool)], dim=1))
            empty = self.model(torch.empty(1, 0, 3), torch.empty(1, 0), 0., torch.empty(1, 0, dtype=torch.bool))
        self.assertEqual(score.shape, (2,))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.allclose(score, padded, atol=1e-6))
        self.assertTrue(torch.allclose(score[1:], empty, atol=1e-6))

    def test_no_future_leakage_forward_and_cap(self):
        x = torch.randn(1, 4, 3)
        days = torch.tensor([[-4., -2., -1., 8.]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        with torch.no_grad():
            first = self.model(x, days, 0., mask)
            x[:, -1] = 1e6
            after = self.model(x, days, 0., mask)
            capped = self.model(x[:, 1:3], days[:, 1:3], 0., mask[:, 1:3])
        self.assertTrue(torch.allclose(first, after, atol=1e-6))
        self.assertTrue(torch.allclose(first, capped, atol=1e-6))

    def test_time_gap_features_after_cap(self):
        captured = []
        handle = self.model.gap_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        self.model(torch.randn(1, 4, 3), torch.tensor([[-100., -4., -1., 5.]]), 0., torch.ones(1, 4, dtype=torch.bool))
        handle.remove()
        gap = captured[0][0, :, 0]
        self.assertAlmostEqual(float(gap[1]), 0.)
        self.assertAlmostEqual(float(gap[2]), np.log1p(3) / np.log1p(self.config.gap_cap_days), places=6)

    def test_sorting_required_and_static_optional(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 2, 3), torch.tensor([[-1., -2.]]), 0., torch.ones(1, 2, dtype=torch.bool))
        static_model = NoteSurvivalTransformer(replace(self.config, static_dim=2)).eval()
        score = static_model(torch.empty(1, 0, 3), torch.empty(1, 0), -.5,
                             torch.empty(1, 0, dtype=torch.bool), torch.tensor([[0., 1.]]))
        self.assertTrue(torch.isfinite(score).all())

    def test_synthetic_signal_loss_decreases_with_stable_gradients(self):
        seed_everything(2)
        model = NoteSurvivalTransformer(small_config(input_dim=3, max_notes=1))
        x = torch.tensor([[[2., 0., 0.]], [[-2., 0., 0.]], [[-1., 0., 0.]]])
        days, mask = torch.full((3, 1), -1.), torch.ones(3, 1, dtype=torch.bool)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
        losses = []
        for _ in range(35):
            optimizer.zero_grad()
            loss = sampled_cox_loss(model(x, days, 0., mask))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(np.mean(losses[-5:]), .5 * losses[0])


class DataAndIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_dir = generate_synthetic(self.root / "data", n_patients=30, input_dim=4)
        self.data = self.load()

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return load_data(self.data_dir / "notes.npy", self.data_dir / "notes.csv",
                         self.data_dir / "outcomes.csv", self.data_dir / "splits.csv",
                         self.data_dir / "patients.csv", embedding_model_id="synthetic-v1")

    def test_split_isolation_and_label_rejection(self):
        splits = pd.read_csv(self.data_dir / "splits.csv")
        pd.concat([splits, splits.iloc[:1]]).to_csv(self.data_dir / "splits.csv", index=False)
        with self.assertRaisesRegex(ValueError, "isolation"):
            self.load()
        splits.to_csv(self.data_dir / "splits.csv", index=False)
        outcomes = pd.read_csv(self.data_dir / "outcomes.csv")
        outcomes.loc[0, ["event", "death_event"]] = 1
        outcomes.to_csv(self.data_dir / "outcomes.csv", index=False)
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.load()
        outcomes.loc[0, "event"] = 0
        outcomes["entry_time"] = 0
        outcomes.to_csv(self.data_dir / "outcomes.csv", index=False)
        with self.assertRaisesRegex(ValueError, "delayed entry"):
            self.load()

    def test_loader_sorting_same_time_and_filter_before_cap(self):
        notes = pd.read_csv(self.data_dir / "notes.csv")
        patient = notes["patient_id"].iloc[0]
        frame = notes[notes.patient_id == patient]
        expected = frame[frame.days_from_index <= .5].sort_values(["days_from_index", "note_id"]).tail(1)
        vectors, days = self.data.history(patient, .5, 1)
        self.assertTrue(np.array_equal(days, expected.days_from_index.to_numpy()))
        self.assertTrue(np.array_equal(vectors, self.data.embeddings[expected.embedding_row]))
        notes.loc[notes.patient_id == patient, "days_from_index"] = -1.
        notes.sample(frac=1, random_state=99).to_csv(self.data_dir / "notes.csv", index=False)
        loaded = self.load()
        selected = loaded.notes[loaded.notes.patient_id == patient]
        self.assertEqual(selected.note_id.tolist(), sorted(selected.note_id))

    def test_input_dimensions_finite_times_and_encoder_provenance(self):
        vectors = np.array(self.data.embeddings, copy=True)
        vectors[0, 0] = np.nan
        np.save(self.data_dir / "notes.npy", vectors)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.load()
        vectors[0, 0] = 0
        np.save(self.data_dir / "notes.npy", vectors)
        notes = pd.read_csv(self.data_dir / "notes.csv")
        notes["embedding_model_id"] = "synthetic-v1"
        notes.loc[0, "embedding_model_id"] = "another-encoder"
        notes.to_csv(self.data_dir / "notes.csv", index=False)
        with self.assertRaisesRegex(ValueError, "common embedding"):
            self.load()

    def test_loader_future_note_invariance_and_train_only_static(self):
        cfg = small_config(input_dim=4)
        static = StaticTransform.fit(self.data.patients, self.data.split_ids("train"))
        cfg.static_dim = static.output_dim
        model = NoteSurvivalTransformer(cfg).eval()
        ids = self.data.split_ids("val")
        batch, _ = self.data.batch(ids, .5, cfg, static)
        with torch.no_grad():
            before = model(**batch)
        vectors = np.array(self.data.embeddings, copy=True)
        future = self.data.notes.loc[self.data.notes.days_from_index > .5, "embedding_row"]
        vectors[future] = 1e5
        np.save(self.data_dir / "notes.npy", vectors)
        changed = self.load()
        with torch.no_grad():
            after = model(**changed.batch(ids, .5, cfg, static)[0])
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        patients = changed.patients.copy()
        patients.loc[~patients.patient_id.isin(changed.split_ids("train")), "baseline_measurement"] = 1e8
        static2 = StaticTransform.fit(patients, changed.split_ids("train"))
        self.assertEqual(static.to_dict(), static2.to_dict())

    def test_checkpoint_reload_outcome_free_inference_and_test_export(self):
        cfg = small_config(num_epochs=1)
        result = fit(self.data, cfg, self.root / "run")
        model, filled, static, state = restore_model(result["checkpoint_path"])
        model2, _, _, _ = restore_model(result["checkpoint_path"])
        batch, _ = self.data.batch(self.data.split_ids("val"), .5, filled, static)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**batch), model2(**batch)))
        outcome_free = load_data(self.data_dir / "notes.npy", self.data_dir / "notes.csv",
                                 patients_path=self.data_dir / "patients.csv", embedding_model_id="synthetic-v1")
        predictions = inference(result["checkpoint_path"], outcome_free, [-1.25, .5])
        self.assertEqual(len(predictions), 60)
        self.assertFalse(predictions.duplicated(["patient_id", "landmark"]).any())
        self.assertNotIn("event", predictions)
        self.assertTrue(np.isfinite(predictions.log_risk_event).all())
        self.assertTrue((predictions.n_notes_used == 0).any())
        final_test_evaluation(result["checkpoint_path"], self.data, self.root / "test")
        frame = pd.read_csv(self.root / "test" / "test_predictions.csv")
        self.assertTrue((frame.time_from_landmark > 0).all())
        self.assertEqual(set(frame.patient_id), set(self.data.split_ids("test")))
        with self.assertRaisesRegex(ValueError, "model/version"):
            outcome_free.embedding_model_id = "different-encoder"
            inference(result["checkpoint_path"], outcome_free)

    def test_runtime_thread_override_preserves_checkpoint_and_test_provenance(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        result = fit(self.data, small_config(num_epochs=1, num_threads=2), self.root / "run")
        checkpoint = Path(result["checkpoint_path"])
        saved_bytes = checkpoint.read_bytes()
        saved_state = load_checkpoint(checkpoint)

        torch.set_num_threads(1)
        _, config, _, state = restore_model(checkpoint)
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual(config.num_threads, 2)
        _, config, _, state = restore_model(checkpoint, num_threads=1)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(config.num_threads, 1)
        self.assertEqual(state["config"], saved_state["config"])

        torch.set_num_threads(2)
        report = final_test_evaluation(checkpoint, self.data, self.root / "test", num_threads=1)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(report["model_id"], saved_state["model_id"])
        self.assertEqual(report["data_fingerprint"], saved_state["data_fingerprint"])
        self.assertTrue(np.isfinite(report["weighted_test_c_index"]))
        predictions = pd.read_csv(self.root / "test" / "test_predictions.csv")
        self.assertEqual(set(predictions.patient_id), set(self.data.split_ids("test")))
        self.assertEqual(checkpoint.read_bytes(), saved_bytes)
        _, config, _, _ = restore_model(checkpoint)
        self.assertEqual(config.num_threads, 2)
        self.assertEqual(torch.get_num_threads(), 2)

    def test_invalid_runtime_thread_override_is_rejected_before_reseeding(self):
        result = fit(self.data, small_config(num_epochs=1), self.root / "run")
        for num_threads in (0, -1, True, 1.5):
            with self.subTest(num_threads=num_threads):
                with patch("pipelines.note_transformer.note_survival_transformer.predict.seed_everything") as seed:
                    with self.assertRaisesRegex(ValueError, "num_threads must be a positive integer"):
                        final_test_evaluation(result["checkpoint_path"], self.data,
                                              self.root / "invalid_test", num_threads=num_threads)
                    seed.assert_not_called()
        self.assertFalse((self.root / "invalid_test").exists())

    def test_microbatch_gradient_is_group_mean(self):
        cfg = small_config(input_dim=4, batch_size=100, weight_decay=0., max_grad_norm=0.)
        static = StaticTransform.fit(None, self.data.split_ids("train"))
        self.data.patients = None
        sampler = RiskSetSampler(self.data.outcomes, self.data.split_ids("train"), cfg.landmarks, 4)
        # A tiny set of groups allows an independent all-graphs-at-once reference.
        sampler.event_indices = sampler.event_indices[:3]
        seed_everything(cfg.seed)
        a = NoteSurvivalTransformer(cfg)
        b = NoteSurvivalTransformer(cfg)
        b.load_state_dict(a.state_dict())
        optimizer_a = torch.optim.AdamW(a.parameters(), lr=cfg.lr, weight_decay=0.)
        optimizer_b = torch.optim.AdamW(b.parameters(), lr=cfg.lr, weight_decay=0.)
        train_epoch(a, self.data, sampler, static, optimizer_a, cfg, 0)
        losses = []
        for group in sampler.epoch_groups(cfg.seed + 1009):
            batch, _ = self.data.batch(group.member_ids, group.landmark, cfg, static)
            losses.append(sampled_cox_loss(b(**batch), group.weights))
        torch.stack(losses).mean().backward()
        optimizer_b.step()
        for name, parameter in a.state_dict().items():
            self.assertTrue(torch.allclose(parameter, b.state_dict()[name], atol=1e-6), name)

    def test_undefined_validation_and_no_eligible_events_fail_explicitly(self):
        self.data.outcomes.loc[self.data.split_ids("val"), "event"] = 0
        with self.assertRaisesRegex(ValueError, "no computable C-index"):
            fit(self.data, small_config(num_epochs=1), self.root / "undefined")
        with self.assertRaisesRegex(ValueError, "No eligible"):
            fit(self.data, small_config(landmarks=(1e6,)), self.root / "no_events")

    def test_epoch_resume_equivalence_and_fingerprint_guard(self):
        cfg = small_config(num_epochs=3, patience=3, dropout=.1, scheduler="cosine")
        uninterrupted = fit(self.data, cfg, self.root / "full")
        paused = fit(self.data, cfg, self.root / "resumed", stop_after_epoch=1)
        self.assertEqual(paused["status"], "interrupted")
        resumed = fit(self.data, cfg, self.root / "resumed", resume=True)
        a, b = load_checkpoint(uninterrupted["checkpoint_path"]), load_checkpoint(resumed["checkpoint_path"])
        self.assertEqual(a["best_validation_score"], b["best_validation_score"])
        for name in a["model_state_dict"]:
            self.assertTrue(torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]), name)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            fit(self.data, replace(cfg, lr=.123), self.root / "resumed", resume=True)

    def test_search_validation_only_resume_and_selection(self):
        cfg = small_config(num_epochs=1)
        space = {"d_model": [8, 10], "n_heads": [2], "lr": [.005, .01]}
        original = evaluate_model
        observed_ids = []
        def validate_only(model, data, patient_ids, *args, **kwargs):
            self.assertEqual(set(patient_ids), set(data.split_ids("val")))
            observed_ids.append(patient_ids)
            return original(model, data, patient_ids, *args, **kwargs)
        with patch("pipelines.note_transformer.note_survival_transformer.train.evaluate_model", side_effect=validate_only):
            summary = run_search(self.data, cfg, space, 2, 19, self.root / "search")
        records = pd.read_csv(self.root / "search" / "hparam_search_results.csv")
        expected = records.sort_values(["validation_score", "trial_id"], ascending=[False, True]).iloc[0]
        self.assertEqual(summary["selected_trial"], expected.trial_id)
        self.assertEqual(summary["validation_score"], expected.validation_score)
        self.assertFalse(summary["test_evaluated"])
        self.assertEqual(len(observed_ids), 2)
        with patch("pipelines.note_transformer.note_survival_transformer.hparam_search.fit", side_effect=AssertionError("Successful trial rerun")):
            again = run_search(self.data, cfg, space, 2, 19, self.root / "search", resume=True)
        self.assertEqual(again["selected_trial"], summary["selected_trial"])
        self.assertTrue((self.root / "search" / "best_config.json").exists())
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            run_search(self.data, cfg, space, 2, 20, self.root / "search", resume=True)

    def test_search_incomplete_trial_resume(self):
        cfg = small_config(num_epochs=2)
        space = {"lr": [.01]}
        search_dir = self.root / "search"
        from pipelines.note_transformer.note_survival_transformer.train import fit as actual_fit
        def interrupt_trial(*args, **kwargs):
            actual_fit(*args, **kwargs, stop_after_epoch=1)
            raise KeyboardInterrupt("simulated interruption")
        with patch("pipelines.note_transformer.note_survival_transformer.hparam_search.fit", side_effect=interrupt_trial):
            with self.assertRaises(KeyboardInterrupt):
                run_search(self.data, cfg, space, 1, 42, search_dir)
        result = run_search(self.data, cfg, space, 1, 42, search_dir, resume=True)
        self.assertEqual(result["successful_trials"], 1)
        self.assertEqual(load_checkpoint(search_dir / "trials" / "trial_0000" / "last_model.pt")["epoch"], 1)

    def test_search_invalid_rejection_and_seeded_distributions(self):
        cfg = small_config()
        space = {"d_model": [8, 9], "n_heads": [2], "lr": {"type": "loguniform", "low": .001, "high": .01},
                 "max_notes": {"type": "int", "low": 2, "high": 4}}
        first, rejected = sample_plan(cfg, space, 5, 42)
        second, _ = sample_plan(cfg, space, 5, 42)
        self.assertEqual([c.to_dict() for c in first], [c.to_dict() for c in second])
        self.assertTrue(all(c.d_model % c.n_heads == 0 for c in first))
        self.assertGreater(len(rejected), 0)
        with self.assertRaisesRegex(ValueError, "compatible"):
            sample_plan(cfg, {"d_model": [9], "n_heads": [2]}, 1, 42)

    def test_partial_undefined_landmarks_are_logged_and_search_resumes(self):
        cfg = small_config(num_epochs=1, landmarks=(-2., 1e6))
        search_dir = self.root / "partial"
        run_search(self.data, cfg, {"lr": [.01]}, 1, 42, search_dir)
        record = json.loads((search_dir / "trials" / "trial_0000" / "trial_result.json").read_text())
        self.assertIsNone(record["landmark_metrics"][1]["c_index"])
        self.assertEqual(record["landmark_metrics"][1]["n_eligible"], 0)
        summary = run_search(self.data, cfg, {"lr": [.01]}, 1, 42, search_dir, resume=True)
        self.assertEqual(summary["successful_trials"], 1)


if __name__ == "__main__":
    unittest.main()
