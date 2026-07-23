from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from wormctx.poc.config import load_config
from wormctx.poc.generation import dataset_summary, generate_configured_episodes
from wormctx.poc.repro import git_manifest, verify_checksums, write_checksums, write_text


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = (
    ROOT
    / "experiments"
    / "graph_conditioned_inference"
    / "synthetic_routing_and_design_selection"
    / "config"
)


class PocConfigReproTests(unittest.TestCase):
    def test_git_manifest_accepts_only_an_exported_commit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "0123456789abcdef" * 2 + "01234567"
            (root / "SOURCE_REVISION").write_text(revision + "\n", encoding="ascii")
            manifest = git_manifest(root)
            self.assertEqual(manifest["commit"], revision)
            self.assertEqual(manifest["provenance"], "git_archive_export_subst")
            self.assertEqual(manifest["status"], "archive_without_git_metadata")

            (root / "SOURCE_REVISION").write_text("$Format:%H$\n", encoding="ascii")
            placeholder_manifest = git_manifest(root)
            self.assertIsNone(placeholder_manifest["commit"])
            self.assertEqual(placeholder_manifest["provenance"], "git_cli")

    def test_bundled_configs_validate(self) -> None:
        smoke = load_config(EXPERIMENT_CONFIG / "smoke.json")
        full = load_config(EXPERIMENT_CONFIG / "full_gpu.json")
        self.assertEqual(smoke.training.device, "auto")
        self.assertEqual(full.training.device, "cuda")
        self.assertEqual(full.model.hidden_dim % full.model.attention_heads, 0)

    def test_configured_smoke_dataset_balances_red_team_labels(self) -> None:
        config = load_config(EXPERIMENT_CONFIG / "smoke.json")
        summary = dataset_summary(generate_configured_episodes(config))
        counts = summary["invalid_flag_counts"]
        self.assertEqual(len(counts), 5)
        self.assertEqual(len(set(counts.values())), 1)

    def test_config_rejects_unknown_fields(self) -> None:
        path = EXPERIMENT_CONFIG / "smoke.json"
        payload = path.read_text(encoding="utf-8").replace(
            '"run_name": "cpu_or_gpu_smoke",',
            '"run_name": "cpu_or_gpu_smoke", "surprise": true,',
        )
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.json"
            bad.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_config(bad)

    def test_checksum_receipt_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_text(root / "result.txt", "original\n")
            write_checksums(root)
            self.assertEqual(verify_checksums(root), [])
            write_text(root / "result.txt", "changed\n")
            self.assertEqual(verify_checksums(root), ["mismatch:result.txt"])


if __name__ == "__main__":
    unittest.main()
