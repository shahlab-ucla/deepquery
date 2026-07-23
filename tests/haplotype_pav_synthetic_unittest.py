"""Standard-library smoke test for runtimes that do not have pytest installed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wormctx.poc import qtl_haplotype_pav as hp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/pangenome_state_qualification/config/pangenome_state_contract.json"


class HaplotypePavSyntheticSmoke(unittest.TestCase):
    def test_qualified_panel_and_plan(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regions = []
            for index, (region_id, (start, end)) in enumerate(
                hp.EXPECTED_REGIONS.items(), 1
            ):
                interval_hash = f"{index:x}" * 64
                regions.append(
                    {
                        "region_id": region_id,
                        "source_assembly": hp.WS276_ASSEMBLY,
                        "target_assembly": hp.WS283_ASSEMBLY,
                        "source_chromosome": 5,
                        "target_chromosome": 5,
                        "source_start": start,
                        "source_end": end,
                        "target_start": start,
                        "target_end": end,
                        "mapping_mode": "sequence_identical_coordinate_identity",
                        "source_interval_sha256": interval_hash,
                        "target_interval_sha256": interval_hash,
                    }
                )
            coordinate_path = root / "coordinate.json"
            coordinate_path.write_text(
                json.dumps(
                    {
                        "schema_version": hp.COORDINATE_RECEIPT_VERSION,
                        "source_assembly": hp.WS276_ASSEMBLY,
                        "target_assembly": hp.WS283_ASSEMBLY,
                        "source_reference_fasta_sha256": "a" * 64,
                        "target_reference_fasta_sha256": "b" * 64,
                        "source_chromosome_v_sha256": "c" * 64,
                        "target_chromosome_v_sha256": "d" * 64,
                        "generated_before_state_ingestion": True,
                        "phenotype_accessed": False,
                        "association_results_accessed": False,
                        "regions": regions,
                    }
                ),
                encoding="utf-8",
            )
            coordinate = hp.qualify_coordinate_identity(manifest, coordinate_path)

            blocks = [
                hp.BlockDefinition(
                    block_id="left_hap_block_01",
                    region_id="chrv_left",
                    chromosome=5,
                    assembly=hp.WS283_ASSEMBLY,
                    start=1_800_000,
                    end=2_100_000,
                    state_kind="haplotype",
                    reference_state_id="hap_ref",
                    definition_method="outcome_blind_pangenome_path_cluster",
                    source_snapshot_sha256="1" * 64,
                    phenotype_accessed_during_definition=False,
                ),
                hp.BlockDefinition(
                    block_id="right_pav_locus_01",
                    region_id="chrv_right",
                    chromosome=5,
                    assembly=hp.WS283_ASSEMBLY,
                    start=15_900_000,
                    end=16_100_000,
                    state_kind="pav",
                    reference_state_id="pav_present",
                    definition_method="outcome_blind_presence_absence_locus",
                    source_snapshot_sha256="2" * 64,
                    phenotype_accessed_during_definition=False,
                ),
            ]
            samples = [f"strain_{index:03d}" for index in range(209)]
            calls: list[hp.GenomicStateCall] = []
            for index, sample in enumerate(samples):
                unresolved = index >= 204
                alternate = 100 <= index < 204
                calls.append(
                    hp.HaplotypeStateCall(
                        sample_id=sample,
                        block_id="left_hap_block_01",
                        region_id="chrv_left",
                        assembly=hp.WS283_ASSEMBLY,
                        state_id=(
                            "hap_unresolved"
                            if unresolved
                            else "hap_alt" if alternate else "hap_ref"
                        ),
                        confidence=0.99,
                        kind="haplotype",
                        state=(
                            "unresolved"
                            if unresolved
                            else "alternate_path" if alternate else "reference_path"
                        ),
                        path_id=(None if unresolved else "path_alt" if alternate else "path_ref"),
                    )
                )
                present = index < 120
                calls.append(
                    hp.PavStateCall(
                        sample_id=sample,
                        block_id="right_pav_locus_01",
                        region_id="chrv_right",
                        assembly=hp.WS283_ASSEMBLY,
                        state_id="pav_present" if present else "pav_absent",
                        confidence=0.98,
                        kind="pav",
                        state="present" if present else "absent",
                        locus_id="pav_locus_01",
                        copy_number=1 if present else 0,
                    )
                )
            panel = hp.qualify_state_panel(manifest, coordinate, blocks, calls, samples)
            self.assertEqual(panel.eligible_block_count, 2)
            self.assertEqual(panel.blocks[0].unresolved_count, 5)

            eligible_cells = [
                cell
                for cell in hp._expected_cell_ids()
                if cell not in hp.EXPECTED_BOUNDARY_CELLS
            ]
            dependency = hp.NullDependencyQualification(
                state="success_descriptive_scaffold_enabled",
                run_id=hp.NULL_RUN_ID,
                terminal_success=True,
                analysis_plan_permitted=True,
                scaffold_development_permitted=True,
                aggregation_sha256="f" * 64,
                interpretation_receipt_sha256=None,
                threshold_eligible_cells=eligible_cells,
                boundary_cells=list(hp.EXPECTED_BOUNDARY_CELLS),
            )
            plan = hp.build_synthetic_analysis_plan(
                manifest, coordinate, dependency, panel
            )
            self.assertEqual(plan.request_count, 32)
            self.assertFalse(plan.real_execution_permitted)
            self.assertAlmostEqual(plan.block_bonferroni_alpha, 0.025)
            self.assertTrue(all(request.descriptive_only for request in plan.requests))
            self.assertTrue(
                all(
                    not request.null_threshold_eligible
                    for request in plan.requests
                    if request.cell_id in hp.EXPECTED_BOUNDARY_CELLS
                )
            )


if __name__ == "__main__":
    unittest.main()
