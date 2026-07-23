from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import wormctx.cli as cli


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_wheel_declares_all_cli_resources(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        included = project["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        self.assertEqual(included["schemas"], "wormctx/resources/schemas")
        self.assertEqual(included["config"], "wormctx/resources/config")
        self.assertEqual(included["experiments"], "wormctx/resources/experiments")
        self.assertEqual(included["mappings"], "wormctx/resources/mappings")
        self.assertEqual(included["sql"], "wormctx/resources/sql")
        self.assertEqual(
            included["data/examples"], "wormctx/resources/data/examples"
        )

    def test_repository_root_falls_back_to_installed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site-packages" / "wormctx"
            resource_root = site / "resources"
            (resource_root / "config").mkdir(parents=True)
            (resource_root / "config" / "source_catalog.json").write_text(
                "{}", encoding="utf-8"
            )
            pretend_module = site / "cli.py"
            with (
                patch.object(cli, "__file__", str(pretend_module)),
                patch.object(cli.resources, "files", return_value=site),
            ):
                self.assertEqual(cli.repository_root(), resource_root)

    def test_context_table_uses_non_nullable_surrogate_key(self) -> None:
        ddl = (ROOT / "sql" / "canonical.sql").read_text(encoding="utf-8")
        table = ddl.split("CREATE TABLE IF NOT EXISTS observation_context_value", 1)[1]
        table = table.split(");", 1)[0]
        self.assertIn("context_value_id VARCHAR PRIMARY KEY", table)
        self.assertNotIn(
            "PRIMARY KEY (observation_id, dimension, value_id, source_value)", table
        )
        self.assertIn("CREATE TABLE IF NOT EXISTS observation_result", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS result_measurement", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS result_comparison_group", ddl)
        self.assertIn("CREATE TABLE IF NOT EXISTS result_asset", ddl)
        self.assertIn("canonical_observation_json JSON NOT NULL", ddl)
        self.assertIn("biolink_knowledge_level VARCHAR", ddl)
        self.assertIn("s.source_release", ddl)
        self.assertIn("s.source_artifact", ddl)


if __name__ == "__main__":
    unittest.main()
