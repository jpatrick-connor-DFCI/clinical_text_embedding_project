"""Focused tests for the optimized Clinical ModernBERT embedding pipeline."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


# The lightweight local test environment may not include zstandard. Its codec is
# immaterial to these unit tests, so use an identity implementation when absent.
if importlib.util.find_spec("zstandard") is None:
    zstandard_stub = types.ModuleType("zstandard")
    zstandard_stub.compress = lambda data, level=3: data
    zstandard_stub.decompress = lambda data: data
    zstandard_stub.ZstdError = RuntimeError
    sys.modules["zstandard"] = zstandard_stub


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipelines"
    / "preprocessing"
    / "generate_clinical_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("generate_clinical_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
embeddings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(embeddings)


class FakeEmbeddingModel(torch.nn.Module):
    """Return token IDs as deterministic two-dimensional hidden states."""

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(last_hidden_state=hidden)


class GenerateClinicalEmbeddingsTests(unittest.TestCase):
    def test_padding_free_collation_builds_flash_attention_metadata(self) -> None:
        packed = embeddings.collate_packed_batch(
            [
                (4, np.array([10, 11], dtype=np.int32)),
                (9, np.array([20, 21, 22], dtype=np.int32)),
            ]
        )
        indices, tokens, position_ids, lengths, cumulative_lengths, max_length = packed

        self.assertEqual(indices.tolist(), [4, 9])
        self.assertEqual(tokens.tolist(), [[10, 11, 20, 21, 22]])
        self.assertEqual(position_ids.tolist(), [[0, 1, 0, 1, 2]])
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(cumulative_lengths.tolist(), [0, 2, 5])
        self.assertEqual(cumulative_lengths.dtype, torch.int32)
        self.assertEqual(max_length, 3)

    def test_packed_embeddings_are_mean_pooled_per_note(self) -> None:
        _, tokens, position_ids, lengths, cumulative_lengths, max_length = (
            embeddings.collate_packed_batch(
                [
                    (0, np.array([2, 4], dtype=np.int64)),
                    (1, np.array([3, 6, 9], dtype=np.int64)),
                ]
            )
        )
        runner = embeddings.ModelRunner(FakeEmbeddingModel(), compile_model=False)
        result = embeddings.embed_minibatch(
            runner,
            tokens,
            position_ids,
            lengths,
            cumulative_lengths,
            max_length,
        )

        torch.testing.assert_close(result, torch.tensor([[3.0, 3.0], [6.0, 6.0]]))

    def test_small_thread_prefetched_embedding_run_restores_note_order(self) -> None:
        token_batch = {
            "input_ids_flat": np.array([10, 2, 4, 1, 3, 5, 7], dtype=np.int64),
            "row_splits": np.array([0, 1, 3, 7], dtype=np.int64),
        }
        runner = embeddings.ModelRunner(FakeEmbeddingModel(), compile_model=False)
        with (
            mock.patch.object(embeddings, "COLLATE_WORKERS", 2),
            mock.patch.object(embeddings, "MAX_BATCH_SIZE", 2),
        ):
            result = embeddings.embed_token_batch(
                runner,
                token_batch,
                hidden_size=2,
                device="cpu",
                pin_memory=False,
                batch_idx=0,
            )

        torch.testing.assert_close(
            result,
            torch.tensor([[10.0, 10.0], [3.0, 3.0], [4.0, 4.0]]),
        )

    def test_attention_budget_uses_sum_of_squared_lengths(self) -> None:
        row_splits = np.array([0, 1, 3, 6], dtype=np.int64)
        with (
            mock.patch.object(embeddings, "MAX_BATCH_SIZE", 10),
            mock.patch.object(embeddings, "MAX_ATTENTION_ELEMENTS", 5),
        ):
            self.assertEqual(embeddings.build_length_bucketed_batches(row_splits), [[0, 1], [2]])

    def test_slice_token_batch_preserves_ragged_boundaries(self) -> None:
        sliced = embeddings.slice_token_batch(
            {
                "input_ids_flat": np.array([1, 2, 3, 4, 5, 6]),
                "row_splits": np.array([0, 2, 3, 6]),
            },
            note_count=2,
        )
        np.testing.assert_array_equal(sliced["input_ids_flat"], [1, 2, 3])
        np.testing.assert_array_equal(sliced["row_splits"], [0, 2, 3])

    def test_atomic_save_ignores_partial_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clinical_notes_embeddings_batch_7.npy.zst"
            expected = np.array([[1.25, 2.5]], dtype=np.float32)
            embeddings.save_embedding_batch(output_path, expected)

            self.assertTrue(output_path.is_file())
            self.assertFalse(output_path.with_suffix(".zst.partial").exists())
            self.assertEqual(embeddings.discover_embedded_batches(Path(temp_dir)), {7})
            raw = embeddings.zstd.decompress(output_path.read_bytes())
            saved = np.load(io.BytesIO(raw), allow_pickle=False)
            np.testing.assert_array_equal(saved, expected.astype(np.float16))

    def test_compile_failure_falls_back_to_eager_once(self) -> None:
        eager_model = FakeEmbeddingModel()

        def failing_compiled_forward(**kwargs: object) -> object:
            raise RuntimeError("unsupported graph")

        with mock.patch.object(embeddings.torch, "compile", return_value=failing_compiled_forward):
            runner = embeddings.ModelRunner(eager_model, compile_model=True)

        result = runner(input_ids=torch.tensor([[5]]))
        self.assertEqual(result.last_hidden_state.tolist(), [[[5.0, 5.0]]])
        self.assertFalse(runner.compiled_active)


if __name__ == "__main__":
    unittest.main()
