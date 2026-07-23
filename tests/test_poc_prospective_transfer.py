from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from wormctx.poc import prospective_transfer as pt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/biological_reasoning/prospective_rnai_transfer/config/prospective_transfer.json"
CANDIDATES = (
    ("eig-1", "graph_eig"),
    ("eig-2", "graph_eig"),
    ("rnd-1", "random"),
    ("rnd-2", "random"),
    ("exp-1", "expert"),
    ("exp-2", "expert"),
    ("unc-1", "uncertainty_only"),
    ("unc-2", "uncertainty_only"),
)
CONTROLS = (
    ("pos-1", "positive_control", "increase"),
    ("pos-2", "positive_control", "decrease"),
    ("neg-1", "negative_control", "near_zero"),
    ("neg-2", "negative_control", "near_zero"),
)
BACKGROUNDS = ("N2", "CB4856", "JU775")
TRUE_OUTCOMES = {
    "eig-1": -0.8,
    "eig-2": -0.6,
    "rnd-1": -0.4,
    "rnd-2": -0.2,
    "exp-1": 0.2,
    "exp-2": 0.4,
    "unc-1": 0.6,
    "unc-2": 0.8,
    "pos-1": 0.9,
    "pos-2": -0.9,
    "neg-1": 0.05,
    "neg-2": -0.05,
}
BACKGROUND_OFFSETS = {"N2": 0.0, "CB4856": 0.1, "JU775": -0.1}
ASSET_BYTES: dict[str, bytes] = {}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _asset(label: str) -> pt.LockedAsset:
    content = label.encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    ASSET_BYTES[digest] = content
    return pt.LockedAsset(
        asset_id=f"asset-{digest[:16]}",
        relative_path=f"{digest}.asset",
        sha256=digest,
        bytes=len(content),
    )


def _materialize_assets(assets: list[pt.LockedAsset], root: Path) -> None:
    root.mkdir(parents=True)
    for asset in assets:
        path = root / asset.relative_path
        path.write_bytes(ASSET_BYTES[asset.sha256])


def _location(gene_id: str) -> str:
    return "AB" if list(TRUE_OUTCOMES).index(gene_id) % 2 == 0 else "E"


def _panel() -> list[pt.GenePanelRecord]:
    records = [
        pt.GenePanelRecord(
            gene_id=gene,
            panel_role="candidate",
            selection_arm=arm,
            candidate_source=f"synthetic:{arm}",
            source_snapshot_sha256=_hash(f"source:{arm}"),
            source_score=float(index + 1) / 10.0,
            expected_outcome_direction="unspecified",
            selected_before_result_access=True,
            sealed_test_metrics_used=False,
        )
        for index, (gene, arm) in enumerate(CANDIDATES)
    ]
    records.extend(
        pt.GenePanelRecord(
            gene_id=gene,
            panel_role=role,
            selection_arm=role,
            candidate_source=f"synthetic:{role}",
            source_snapshot_sha256=_hash(f"source:{role}"),
            source_score=None,
            expected_outcome_direction=direction,
            selected_before_result_access=True,
            sealed_test_metrics_used=False,
        )
        for gene, role, direction in CONTROLS
    )
    return records


def _backgrounds() -> list[pt.BackgroundRecord]:
    return [
        pt.BackgroundRecord(
            background_id="N2",
            role="reference",
            selection_source="synthetic reference",
            genotype_snapshot_sha256=_hash("genotype:N2"),
            haplotype_panel_receipt_sha256=None,
            selected_before_result_access=True,
        ),
        *[
            pt.BackgroundRecord(
                background_id=background,
                role="natural_haplotype",
                selection_source="synthetic haplotype diversity panel",
                genotype_snapshot_sha256=_hash(f"genotype:{background}"),
                haplotype_panel_receipt_sha256=_hash(f"haplotype:{background}"),
                selected_before_result_access=True,
            )
            for background in BACKGROUNDS[1:]
        ],
    ]


def _forecast_records(
    regime: str, target_cells: set[tuple[str, str]]
) -> list[pt.ForecastRecord]:
    regime_error = {"zero_shot": 0.20, "low_shot": 0.10, "retrained": 0.02}
    forecasts = []
    for gene, background in sorted(target_cells):
        truth = TRUE_OUTCOMES[gene] + BACKGROUND_OFFSETS[background]
        for architecture in pt.ARCHITECTURES:
            abstain = (
                architecture == "edge_free"
                and gene == "unc-2"
                and background == "JU775"
            )
            error = regime_error[regime] if architecture == "graph" else 0.60
            forecasts.append(
                pt.ForecastRecord(
                    gene_id=gene,
                    background_id=background,
                    architecture=architecture,
                    transfer_regime=regime,
                    model_artifact_sha256=_hash(f"model:{architecture}:{regime}"),
                    predicted_later_outcome=None if abstain else truth + error,
                    interval_lower=None if abstain else truth + error - 0.30,
                    interval_upper=None if abstain else truth + error + 0.30,
                    scalar_uncertainty=0.9 if abstain else abs(error),
                    lineage_location=(
                        None
                        if abstain
                        else _location(gene) if architecture == "graph" else "MS"
                    ),
                    abstain=abstain,
                    abstention_reason="outside frozen support" if abstain else None,
                    frozen_before_test_result_access=True,
                )
            )
    return forecasts


def _plate_assignments() -> list[pt.PlateAssignment]:
    assignments = []
    blind_index = 0
    for background in BACKGROUNDS:
        for replicate in range(1, pt.REPLICATES + 1):
            for well_index, gene in enumerate(TRUE_OUTCOMES, start=1):
                blind_index += 1
                assignments.append(
                    pt.PlateAssignment(
                        blinded_sample_id=f"blind-{blind_index:03d}",
                        gene_id=gene,
                        background_id=background,
                        biological_replicate=replicate,
                        plate_id=f"plate-{background}-{replicate}",
                        well_id=f"A{well_index:02d}",
                        scorer_mapping_visible=False,
                    )
                )
    return assignments


def _transfer_evaluations() -> list[pt.TransferEvaluation]:
    genes = list(TRUE_OUTCOMES)
    candidates = [gene for gene, _ in CANDIDATES]
    controls = [gene for gene, _, _ in CONTROLS]
    n2_cells = [pt._cell(gene, "N2") for gene in genes]
    natural_cells = [
        pt._cell(gene, background)
        for gene in genes
        for background in BACKGROUNDS[1:]
    ]
    low_adaptation = [
        pt._cell(gene, background)
        for gene in controls
        for background in BACKGROUNDS[1:]
    ]
    low_test = [
        pt._cell(gene, background)
        for gene in candidates
        for background in BACKGROUNDS[1:]
    ]
    retrained_test = [
        pt._cell(gene, BACKGROUNDS[1 + index % 2])
        for index, gene in enumerate(candidates)
    ]
    return [
        pt.TransferEvaluation(
            regime="zero_shot",
            train_cells=n2_cells,
            adaptation_cells=[],
            test_cells=natural_cells,
            maximum_adaptation_replicates_per_cell=0,
            test_cells_used_for_training_or_adaptation=False,
        ),
        pt.TransferEvaluation(
            regime="low_shot",
            train_cells=n2_cells,
            adaptation_cells=low_adaptation,
            test_cells=low_test,
            maximum_adaptation_replicates_per_cell=1,
            test_cells_used_for_training_or_adaptation=False,
        ),
        pt.TransferEvaluation(
            regime="retrained",
            train_cells=n2_cells,
            adaptation_cells=sorted(set(natural_cells) - set(retrained_test)),
            test_cells=retrained_test,
            maximum_adaptation_replicates_per_cell=3,
            test_cells_used_for_training_or_adaptation=False,
        ),
    ]


def _preregistration(manifest: pt.ProspectiveTransferManifest) -> pt.Preregistration:
    labels = {
        *(f"source:{arm}" for _, arm in CANDIDATES),
        *(f"source:{role}" for _, role, _ in CONTROLS),
        "random seed",
        "random universe",
        "expert panel",
        "uncertainty scores",
        "graph eig scores",
        *(f"genotype:{background}" for background in BACKGROUNDS),
        *(f"haplotype:{background}" for background in BACKGROUNDS[1:]),
        *(f"model:{architecture}:zero_shot" for architecture in pt.ARCHITECTURES),
        "RNAi protocol",
        "plate randomization seed",
        "power assumptions",
    }
    all_cells = {
        (gene, background) for gene in TRUE_OUTCOMES for background in BACKGROUNDS
    }
    return pt.Preregistration(
        schema_version=pt.PREREG_VERSION,
        preregistration_id="synthetic-prospective_transfer-prereg-v1",
        created_utc="2026-07-21T00:00:00Z",
        synthetic_fixture=True,
        external_model_calls_performed=False,
        result_data_accessed=False,
        referenced_assets=[_asset(label) for label in sorted(labels)],
        panel=_panel(),
        selection_assignment_receipt=pt.SelectionAssignmentReceipt(
            random_seed_sha256=_hash("random seed"),
            random_candidate_universe_sha256=_hash("random universe"),
            expert_panel_receipt_sha256=_hash("expert panel"),
            uncertainty_score_snapshot_sha256=_hash("uncertainty scores"),
            graph_eig_score_snapshot_sha256=_hash("graph eig scores"),
            assignments_created_before_result_access=True,
            result_data_used_for_assignment=False,
        ),
        backgrounds=_backgrounds(),
        forecasts=_forecast_records("zero_shot", all_cells),
        assay=pt.AssayPlan(
            protocol_snapshot_sha256=_hash("RNAi protocol"),
            intervention="RNAi",
            delivery_method="synthetic feeding protocol placeholder",
            dose="frozen synthetic dose placeholder",
            exposure_window="frozen synthetic developmental window placeholder",
            endpoint="later_reporter_deviation_standardized",
            minimum_embryos_per_replicate=20,
            biological_replicates=3,
            randomization_seed_sha256=_hash("plate randomization seed"),
            operator_blinded_to_forecasts=True,
            scorer_blinded_to_gene_background_and_model=True,
            sample_mapping_access="unblinded_coordinator_only",
            unblinding_after_result_lock=True,
            power=pt.PowerPlan(
                assumptions_sha256=_hash("power assumptions"),
                method="clustered_simulation_frozen_before_results",
                family_alpha=0.05,
                target_power=0.8,
                minimum_detectable_standardized_effect=0.5,
                intraclass_correlation=0.1,
                biological_replicates=3,
                embryos_per_replicate=20,
            ),
        ),
        plate_assignments=_plate_assignments(),
        transfer_evaluations=_transfer_evaluations(),
        independent_dataset_alternative=None,
        result_interpretation_contract_sha256=pt.interpretation_contract_sha256(manifest),
    )


def _adaptation_receipt(
    manifest: pt.ProspectiveTransferManifest,
    preregistration: pt.Preregistration,
    preregistration_bundle_sha256: str,
    blinded_result_lock_bundle_sha256: str,
    stage: str,
) -> pt.AdaptationForecastReceipt:
    evaluation = next(
        item for item in preregistration.transfer_evaluations if item.regime == stage
    )
    test_cells = {tuple(item.split("::", 1)[::-1]) for item in evaluation.test_cells}
    labels = {"training:" + stage, *(f"model:{item}:{stage}" for item in pt.ARCHITECTURES)}
    scope = {
        "low_shot": "post_control_adaptation_transfer_benchmark_nonprospective",
        "retrained": "cross_fitted_post_outcome_transfer_benchmark_nonprospective",
    }[stage]
    return pt.AdaptationForecastReceipt(
        schema_version=pt.ADAPTATION_VERSION,
        adaptation_id=f"synthetic-{stage}-v1",
        created_utc=(
            "2026-07-21T01:00:00Z"
            if stage == "low_shot"
            else "2026-07-21T02:00:00Z"
        ),
        stage=stage,
        status="frozen_after_permitted_adaptation_before_test_unblinding",
        interpretation_scope=scope,
        preregistration_bundle_sha256=preregistration_bundle_sha256,
        blinded_result_lock_bundle_sha256=blinded_result_lock_bundle_sha256,
        manifest_sha256=pt.manifest_sha256(manifest),
        synthetic_fixture=True,
        external_model_calls_performed=False,
        train_cells=evaluation.train_cells,
        adaptation_cells=evaluation.adaptation_cells,
        test_cells=evaluation.test_cells,
        outcome_cells_accessed=sorted(
            set(evaluation.train_cells + evaluation.adaptation_cells)
        ),
        outcome_replicates_accessed={
            **{cell: pt.REPLICATES for cell in evaluation.train_cells},
            **{
                cell: evaluation.maximum_adaptation_replicates_per_cell
                for cell in evaluation.adaptation_cells
            },
        },
        test_outcomes_accessed=False,
        training_data_lock_sha256=_hash(f"training:{stage}"),
        forecasts=_forecast_records(stage, test_cells),
        referenced_assets=[_asset(label) for label in sorted(labels)],
    )


def _blinded_result_lock_fixture(
    root: Path,
    manifest: pt.ProspectiveTransferManifest,
    preregistration: pt.Preregistration,
    preregistration_hash: str,
    observations: list[pt.ResultObservation] | None = None,
) -> tuple[pt.BlindedResultLock, Path]:
    raw_root = root / "raw"
    raw_root.mkdir()
    raw_path = raw_root / "synthetic_measurements.csv"
    raw_path.write_bytes(b"synthetic blinded measurement fixture\n")
    return (
        pt.BlindedResultLock(
            schema_version=pt.BLINDED_RESULT_LOCK_VERSION,
            lock_id="synthetic-blinded-result-lock-v1",
            locked_utc="2026-07-21T00:30:00Z",
            preregistration_bundle_sha256=preregistration_hash,
            manifest_sha256=pt.manifest_sha256(manifest),
            collected_before_any_partial_unblinding=True,
            scorer_blinded=True,
            sample_mapping_released=False,
            synthetic_fixture=True,
            external_experiment_performed=False,
            observations=(
                observations
                if observations is not None
                else _observations(preregistration)
            ),
            raw_assets=[
                pt.RawAsset(
                    asset_id="synthetic-measurements",
                    relative_path="synthetic_measurements.csv",
                    sha256=pt._sha256(raw_path),
                    bytes=raw_path.stat().st_size,
                )
            ],
        ),
        raw_root,
    )


def _freeze_protocol(
    root: Path,
    manifest: pt.ProspectiveTransferManifest,
    preregistration: pt.Preregistration,
    observations: list[pt.ResultObservation] | None = None,
) -> tuple[Path, Path, Path, dict[str, Path], dict[str, str], str]:
    prereg_assets = root / "prereg-assets"
    _materialize_assets(preregistration.referenced_assets, prereg_assets)
    prereg_root = root / "preregistration"
    prereg_published = pt.freeze_preregistration(
        manifest, preregistration, prereg_assets, prereg_root
    )
    locked_results, raw_root = _blinded_result_lock_fixture(
        root,
        manifest,
        preregistration,
        prereg_published["bundle_sha256"],
        observations,
    )
    blinded_result_lock_root = root / "blinded-result-lock"
    blinded_published = pt.freeze_blinded_result_lock(
        manifest,
        prereg_root,
        locked_results,
        raw_root,
        blinded_result_lock_root,
    )
    stage_roots = {}
    stage_hashes = {}
    for stage in ("low_shot", "retrained"):
        receipt = _adaptation_receipt(
            manifest,
            preregistration,
            prereg_published["bundle_sha256"],
            blinded_published["bundle_sha256"],
            stage,
        )
        asset_root = root / f"{stage}-assets"
        _materialize_assets(receipt.referenced_assets, asset_root)
        stage_root = root / f"{stage}-bundle"
        published = pt.freeze_adaptation_forecasts(
            manifest,
            prereg_root,
            blinded_result_lock_root,
            receipt,
            asset_root,
            stage_root,
        )
        stage_roots[stage] = stage_root
        stage_hashes[stage] = published["bundle_sha256"]
    return (
        prereg_root,
        blinded_result_lock_root,
        raw_root,
        stage_roots,
        stage_hashes,
        blinded_published["bundle_sha256"],
    )


def _observations(
    preregistration: pt.Preregistration,
    underpowered_cell: tuple[str, str] | None = None,
) -> list[pt.ResultObservation]:
    replicate_error = {1: -0.02, 2: 0.0, 3: 0.02}
    observations = []
    for item in preregistration.plate_assignments:
        excluded = (
            underpowered_cell == (item.gene_id, item.background_id)
            and item.biological_replicate in (1, 2)
        )
        observations.append(
            pt.ResultObservation(
                blinded_sample_id=item.blinded_sample_id,
                embryos_measured=0 if excluded else 24,
                observed_later_outcome=(
                    None
                    if excluded
                    else TRUE_OUTCOMES[item.gene_id]
                    + BACKGROUND_OFFSETS[item.background_id]
                    + replicate_error[item.biological_replicate]
                ),
                observed_lineage_location=(None if excluded else _location(item.gene_id)),
                qc_pass=not excluded,
                exclusion_code="technical_failure" if excluded else None,
            )
        )
    return observations


def _write_result_input(
    root: Path,
    preregistration_hash: str,
    blinded_result_lock_root: Path,
    blinded_result_lock_hash: str,
    adaptation_hashes: dict[str, str],
) -> tuple[Path, pt.ResultInput]:
    locked_results = pt.BlindedResultLock.model_validate_json(
        (blinded_result_lock_root / "blinded_result_lock.json").read_text(
            encoding="utf-8"
        )
    )
    results = pt.ResultInput(
        schema_version=pt.RESULT_INPUT_VERSION,
        preregistration_bundle_sha256=preregistration_hash,
        blinded_result_lock_bundle_sha256=blinded_result_lock_hash,
        adaptation_bundle_sha256=adaptation_hashes,
        result_locked_utc="2026-07-21T03:00:00Z",
        collected_before_unblinding=True,
        scorer_blinded=True,
        synthetic_fixture=True,
        external_experiment_performed=False,
        observations=locked_results.observations,
        raw_assets=locked_results.raw_assets,
    )
    result_path = root / "result_input.json"
    result_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    return result_path, results


class ProspectiveTransferTests(unittest.TestCase):
    def test_manifest_preregistration_and_single_lineage_contract(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        preregistration = _preregistration(manifest)
        qualified = pt.validate_preregistration(manifest, preregistration)
        self.assertEqual(qualified["panel_genes"], 12)
        self.assertEqual(qualified["forecasts"], 72)
        self.assertEqual(qualified["expected_observations"], 108)
        self.assertEqual(qualified["plates"], 9)
        self.assertFalse(qualified["biological_claims_permitted"])

        payload = preregistration.forecasts[0].model_dump()
        payload["lineage_location"] = ["AB", "E"]
        with self.assertRaises(ValidationError):
            pt.ForecastRecord.model_validate(payload)

        aggregate_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        aggregate_payload["assay"]["control_gate_level"] = "background_aggregate"
        aggregate_manifest = pt.ProspectiveTransferManifest.model_validate(
            aggregate_payload
        )
        self.assertEqual(
            aggregate_manifest.assay.control_gate_level, "background_aggregate"
        )

    def test_empty_low_shot_and_n2_retrained_holdouts_are_refused(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        preregistration = _preregistration(manifest)

        empty_low = preregistration.model_copy(deep=True)
        empty_low.transfer_evaluations[1].train_cells = []
        with self.assertRaisesRegex(ValueError, "low-shot"):
            pt.validate_preregistration(manifest, empty_low)

        invalid_retrained = preregistration.model_copy(deep=True)
        invalid_retrained.transfer_evaluations[2].train_cells = []
        invalid_retrained.transfer_evaluations[2].test_cells = [
            pt._cell(gene, "N2") for gene, _ in CANDIDATES
        ]
        with self.assertRaisesRegex(ValueError, "natural-background"):
            pt.validate_preregistration(manifest, invalid_retrained)

    def test_file_backed_preregistration_and_manifest_seal(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        preregistration = _preregistration(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            _materialize_assets(preregistration.referenced_assets, assets)
            output = root / "preregistration"
            published = pt.freeze_preregistration(
                manifest, preregistration, assets, output
            )
            self.assertEqual(published["payload_count"], 5)
            self.assertEqual(pt.verify_preregistration(output, manifest)["payload_count"], 5)
            changed_manifest = manifest.model_copy(deep=True)
            changed_manifest.claims.permitted.append("post-freeze mutation")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                pt.verify_preregistration(output, changed_manifest)
            with self.assertRaises(FileExistsError):
                pt.freeze_preregistration(manifest, preregistration, assets, output)

            bad_assets = root / "bad-assets"
            _materialize_assets(preregistration.referenced_assets, bad_assets)
            first = bad_assets / preregistration.referenced_assets[0].relative_path
            first.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum or byte count"):
                pt.freeze_preregistration(
                    manifest, preregistration, bad_assets, root / "bad-prereg"
                )

    def test_staged_adaptation_seals_refuse_test_outcome_access(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        preregistration = _preregistration(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg_assets = root / "prereg-assets"
            _materialize_assets(preregistration.referenced_assets, prereg_assets)
            prereg_root = root / "prereg"
            frozen = pt.freeze_preregistration(
                manifest, preregistration, prereg_assets, prereg_root
            )
            early_receipt = _adaptation_receipt(
                manifest,
                preregistration,
                frozen["bundle_sha256"],
                "0" * 64,
                "low_shot",
            )
            early_assets = root / "early-adaptation-assets"
            _materialize_assets(early_receipt.referenced_assets, early_assets)
            with self.assertRaises(FileNotFoundError):
                pt.freeze_adaptation_forecasts(
                    manifest,
                    prereg_root,
                    root / "missing-blinded-result-lock",
                    early_receipt,
                    early_assets,
                    root / "early-adaptation",
                )

            locked_results, raw_root = _blinded_result_lock_fixture(
                root, manifest, preregistration, frozen["bundle_sha256"]
            )
            blinded_root = root / "blinded-result-lock"
            blinded = pt.freeze_blinded_result_lock(
                manifest, prereg_root, locked_results, raw_root, blinded_root
            )
            self.assertEqual(blinded["payload_count"], 3)
            self.assertEqual(
                pt.verify_blinded_result_lock(
                    blinded_root, manifest, prereg_root
                )["bundle_sha256"],
                blinded["bundle_sha256"],
            )
            receipt = _adaptation_receipt(
                manifest,
                preregistration,
                frozen["bundle_sha256"],
                blinded["bundle_sha256"],
                "low_shot",
            )
            leaked = receipt.model_copy(deep=True)
            leaked.outcome_cells_accessed[-1] = leaked.test_cells[0]
            with self.assertRaisesRegex(ValueError, "outcome access"):
                pt.validate_adaptation_receipt(
                    manifest,
                    preregistration,
                    frozen["bundle_sha256"],
                    blinded["bundle_sha256"],
                    locked_results,
                    leaked,
                )

            wrong_binding = receipt.model_copy(deep=True)
            wrong_binding.blinded_result_lock_bundle_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "different blinded-result lock"):
                pt.validate_adaptation_receipt(
                    manifest,
                    preregistration,
                    frozen["bundle_sha256"],
                    blinded["bundle_sha256"],
                    locked_results,
                    wrong_binding,
                )

            assets = root / "adaptation-assets"
            _materialize_assets(receipt.referenced_assets, assets)
            with self.assertRaisesRegex(ValueError, "different blinded-result lock"):
                pt.freeze_adaptation_forecasts(
                    manifest,
                    prereg_root,
                    blinded_root,
                    wrong_binding,
                    assets,
                    root / "wrong-lock-adaptation",
                )
            published = pt.freeze_adaptation_forecasts(
                manifest,
                prereg_root,
                blinded_root,
                receipt,
                assets,
                root / "adaptation",
            )
            self.assertEqual(published["stage"], "low_shot")
            self.assertEqual(published["payload_count"], 3)
            first_asset = assets / receipt.referenced_assets[0].relative_path
            first_asset.write_bytes(b"tampered adaptation asset")
            with self.assertRaisesRegex(ValueError, "checksum or byte count"):
                pt.freeze_adaptation_forecasts(
                    manifest,
                    prereg_root,
                    blinded_root,
                    receipt,
                    assets,
                    root / "tampered-adaptation",
                )

            (blinded_root / "blinded_result_lock.json").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                pt.verify_blinded_result_lock(blinded_root, manifest, prereg_root)

    def test_full_closure_scores_common_universe_uncertainty_and_transfer(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        preregistration = _preregistration(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                prereg_root,
                blinded_root,
                raw_root,
                stages,
                stage_hashes,
                blinded_hash,
            ) = _freeze_protocol(root, manifest, preregistration)
            prereg_hash = pt.verify_preregistration(
                prereg_root, manifest
            )["bundle_sha256"]
            result_path, results = _write_result_input(
                root, prereg_hash, blinded_root, blinded_hash, stage_hashes
            )
            result_root = root / "result-bundle"
            published = pt.ingest_results(
                manifest,
                prereg_root,
                blinded_root,
                stages,
                result_path,
                raw_root,
                result_root,
            )
            self.assertEqual(published["interpretation_branch"], "graph_advantage_supported")
            summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["controls"]["pass"])
            self.assertTrue(all(
                item["pass"] for item in summary["controls"]["by_background"].values()
            ))
            self.assertEqual(summary["controls"]["gate_level"], "strict_per_plate")
            self.assertTrue(summary["controls"]["plate_gate_pass"])
            self.assertEqual(len(summary["controls"]["by_plate"]), 9)
            self.assertEqual(summary["graph"]["target_cells"], 8)
            self.assertIsNotNone(summary["graph"]["interval_coverage"])
            self.assertIsNotNone(summary["graph"]["uncertainty_calibration_mae"])
            self.assertEqual(
                summary["paired_transfer_comparisons"]["zero_to_low_shot"][
                    "test_cells"
                ],
                16,
            )
            decision = summary["transfer_decision"]
            self.assertTrue(decision["low_shot_nonprospective_improvement_supported"])
            self.assertTrue(decision["retrained_cross_fitted_improvement_supported"])
            self.assertEqual(pt.verify_result_bundle(result_root)["payload_count"], 4)

            changed_manifest = manifest.model_copy(deep=True)
            changed_manifest.claims.permitted.append("post-freeze mutation")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                pt.ingest_results(
                    changed_manifest,
                    prereg_root,
                    blinded_root,
                    stages,
                    result_path,
                    raw_root,
                    root / "changed-manifest-result",
                )

            missing = results.model_copy(deep=True)
            missing.observations.pop()
            missing_path = root / "missing-result.json"
            missing_path.write_text(missing.model_dump_json(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete blinded raw/QC lock"):
                pt.ingest_results(
                    manifest,
                    prereg_root,
                    blinded_root,
                    stages,
                    missing_path,
                    raw_root,
                    root / "missing-result-bundle",
                )

            wrong_blinded_binding = results.model_copy(deep=True)
            wrong_blinded_binding.blinded_result_lock_bundle_sha256 = "0" * 64
            wrong_blinded_path = root / "wrong-blinded-binding.json"
            wrong_blinded_path.write_text(
                wrong_blinded_binding.model_dump_json(), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "different blinded-result lock"):
                pt.ingest_results(
                    manifest,
                    prereg_root,
                    blinded_root,
                    stages,
                    wrong_blinded_path,
                    raw_root,
                    root / "wrong-blinded-binding-result",
                )

            raw_path = raw_root / results.raw_assets[0].relative_path
            raw_path.write_bytes(b"tampered raw result")
            with self.assertRaisesRegex(ValueError, "checksum or byte count"):
                pt.ingest_results(
                    manifest,
                    prereg_root,
                    blinded_root,
                    stages,
                    result_path,
                    raw_root,
                    root / "tampered-raw-result",
                )

    def test_transfer_improvement_requires_common_answered_cell_coverage(self) -> None:
        target_cells = {
            (gene, background)
            for gene, _ in CANDIDATES
            for background in BACKGROUNDS[1:]
        }
        ordered = sorted(target_cells)
        earlier = [
            item
            for item in _forecast_records("zero_shot", target_cells)
            if item.architecture == "graph"
        ]
        later = [
            item
            for item in _forecast_records("low_shot", target_cells)
            if item.architecture == "graph"
        ]

        def abstain_on(
            forecasts: list[pt.ForecastRecord], cells: set[tuple[str, str]]
        ) -> list[pt.ForecastRecord]:
            changed = []
            for forecast in forecasts:
                if (forecast.gene_id, forecast.background_id) in cells:
                    payload = forecast.model_dump()
                    payload.update(
                        {
                            "predicted_later_outcome": None,
                            "interval_lower": None,
                            "interval_upper": None,
                            "lineage_location": None,
                            "abstain": True,
                            "abstention_reason": "synthetic divergent abstention",
                        }
                    )
                    forecast = pt.ForecastRecord.model_validate(payload)
                changed.append(forecast)
            return changed

        earlier = abstain_on(earlier, set(ordered[:4]))
        later = abstain_on(later, set(ordered[-4:]))
        observed = {
            cell: (
                TRUE_OUTCOMES[cell[0]] + BACKGROUND_OFFSETS[cell[1]],
                _location(cell[0]),
            )
            for cell in target_cells
        }
        self.assertEqual(pt._forecast_metrics(earlier, observed, target_cells)["coverage"], 0.75)
        self.assertEqual(pt._forecast_metrics(later, observed, target_cells)["coverage"], 0.75)
        comparison = pt._paired_regime_comparison(
            earlier, later, observed, target_cells, 0.75, 0.05
        )
        self.assertEqual(comparison["common_coverage"], 0.5)
        self.assertEqual(comparison["status"], "insufficient_common_answered_coverage")
        self.assertIsNone(comparison["rmse_improvement"])
        self.assertIsNone(comparison["supported"])

    def test_strict_plate_control_gate_catches_failure_hidden_by_aggregation(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        self.assertEqual(manifest.assay.control_gate_level, "strict_per_plate")
        preregistration = _preregistration(manifest)
        observations = _observations(preregistration)
        assignments = {
            item.blinded_sample_id: item for item in preregistration.plate_assignments
        }
        failed_plate = None
        for observation in observations:
            assignment = assignments[observation.blinded_sample_id]
            if (
                assignment.gene_id == "pos-1"
                and assignment.background_id == "N2"
                and assignment.biological_replicate == 1
            ):
                observation.observed_later_outcome = 0.0
                failed_plate = assignment.plate_id
                break
        self.assertIsNotNone(failed_plate)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                prereg_root,
                blinded_root,
                raw_root,
                stages,
                stage_hashes,
                blinded_hash,
            ) = _freeze_protocol(root, manifest, preregistration, observations)
            prereg_hash = pt.verify_preregistration(
                prereg_root, manifest
            )["bundle_sha256"]
            result_path, _ = _write_result_input(
                root, prereg_hash, blinded_root, blinded_hash, stage_hashes
            )
            result_root = root / "strict-plate-result"
            published = pt.ingest_results(
                manifest,
                prereg_root,
                blinded_root,
                stages,
                result_path,
                raw_root,
                result_root,
            )
            self.assertEqual(
                published["interpretation_branch"], "control_failure_inconclusive"
            )
            summary = json.loads(
                (result_root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["controls"]["background_aggregate_pass"])
            self.assertTrue(summary["controls"]["by_background"]["N2"]["pass"])
            self.assertFalse(summary["controls"]["plate_gate_pass"])
            self.assertFalse(summary["controls"]["by_plate"][failed_plate]["pass"])
            self.assertFalse(summary["controls"]["pass"])

    def test_abstain_underpower_and_control_reversal_close_inconclusively(self) -> None:
        manifest = pt.load_manifest(CONFIG)
        for mode in ("all_abstain", "underpowered", "control_reversed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                preregistration = _preregistration(manifest)
                if mode == "all_abstain":
                    replaced = []
                    candidate_ids = {gene for gene, _ in CANDIDATES}
                    for forecast in preregistration.forecasts:
                        if forecast.background_id == "N2" and forecast.gene_id in candidate_ids:
                            payload = forecast.model_dump()
                            payload.update(
                                {
                                    "predicted_later_outcome": None,
                                    "interval_lower": None,
                                    "interval_upper": None,
                                    "lineage_location": None,
                                    "abstain": True,
                                    "abstention_reason": "synthetic no-support case",
                                }
                            )
                            forecast = pt.ForecastRecord.model_validate(payload)
                        replaced.append(forecast)
                    preregistration.forecasts = replaced
                observations = _observations(
                    preregistration,
                    underpowered_cell=("eig-1", "CB4856")
                    if mode == "underpowered"
                    else None,
                )
                if mode == "control_reversed":
                    assignments = {
                        item.blinded_sample_id: item
                        for item in preregistration.plate_assignments
                    }
                    for observation in observations:
                        if assignments[observation.blinded_sample_id].gene_id == "pos-1":
                            observation.observed_later_outcome = -0.9
                (
                    prereg_root,
                    blinded_root,
                    raw_root,
                    stages,
                    stage_hashes,
                    blinded_hash,
                ) = _freeze_protocol(root, manifest, preregistration, observations)
                prereg_hash = pt.verify_preregistration(
                    prereg_root, manifest
                )["bundle_sha256"]
                result_path, _ = _write_result_input(
                    root, prereg_hash, blinded_root, blinded_hash, stage_hashes
                )
                result_root = root / "result"
                published = pt.ingest_results(
                    manifest,
                    prereg_root,
                    blinded_root,
                    stages,
                    result_path,
                    raw_root,
                    result_root,
                )
                expected = (
                    "insufficient_forecast_coverage_inconclusive"
                    if mode == "all_abstain"
                    else "insufficient_data_inconclusive"
                    if mode == "underpowered"
                    else "control_failure_inconclusive"
                )
                self.assertEqual(published["interpretation_branch"], expected)
                summary = json.loads(
                    (result_root / "summary.json").read_text(encoding="utf-8")
                )
                if mode == "all_abstain":
                    self.assertIsNone(summary["graph"]["rmse"])
                    self.assertEqual(summary["graph"]["status"], "all_abstained_or_unavailable")
                else:
                    if mode == "underpowered":
                        self.assertIn(
                            "CB4856::eig-1",
                            summary["missing_candidate_or_test_cells"],
                        )
                    else:
                        self.assertFalse(
                            summary["controls"]["by_background"]["N2"]["pass"]
                        )


if __name__ == "__main__":
    unittest.main()
