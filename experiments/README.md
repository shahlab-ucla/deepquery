# Experiment catalog

Every experiment has a scientific question, a frozen or example contract, explicit controls,
an implementation module, and a claim boundary. Status values distinguish completed
diagnostics from synthetic qualification, active work, and blocked real-data execution.

| Experiment | Primary question | Current public status | Implementation |
|---|---|---|---|
| [Context graph and transport smoke](context_graph/context_graph_transport_smoke/) | Can the rights-safe graph and transport fixture execute through the portable receipt layer? | Executable portability control | `wormctx.pipeline`, `wormctx.execution` |
| [Synthetic routing and design selection](graph_conditioned_inference/synthetic_routing_and_design_selection/) | Can one shared graph encoder route typed biological inference tasks and reject invalid designs? | Executable synthetic benchmark | `wormctx.poc.model`, `operators`, `training` |
| [Historical marker identity](natural_variation/historical_marker_identity/) | Do recovered historical assets reproduce the deposited marker universe? | Reproduction blocked by source/preprocessing mismatch | `qtl_ws276`, `qtl_ws276_sensitivity` |
| [Modern SNP association reanalysis](natural_variation/modern_snp_association_reanalysis/) | Can a four-trait modern association path be made provenance-bound and auditable? | Technical reanalysis complete; biological claims prohibited | `qtl` |
| [Population-structure sensitivity](natural_variation/population_structure_sensitivity/) | How do fixed principal components alter association calibration? | PC0/PC3/PC5/PC10 sensitivity complete | `qtl_calibration` |
| [Chromosome-excluded relationship sensitivity](natural_variation/chromosome_excluded_relationship_sensitivity/) | Does chromosome-excluded, LD-pruned relatedness reduce inflation? | Six matrices, 48 fits, and eight maps complete | `qtl_grm_sensitivity` |
| [Parametric polygenic-null calibration](natural_variation/parametric_polygenic_null_calibration/) | Does the fitted analysis control error under explicit polygenic nulls? | 1,600 maps complete; calibration is model-dependent | `qtl_parametric_null` |
| [Matched relationship-model diagnostic](natural_variation/matched_relationship_model_diagnostic/) | Is generator/fitter mismatch responsible for null inflation? | 400 paired maps complete; matched model is conservative | `qtl_matched_relationship_model_diagnostic` |
| [Restricted-residual bootstrap calibration](natural_variation/restricted_residual_bootstrap_calibration/) | Can covariance-preserving residual permutations calibrate the intended model? | Smoke qualification complete; terminal full result not public | `qtl_restricted_residual_bootstrap` |
| [Reference coordinate identity](natural_variation/reference_coordinate_identity/) | Are frozen regions coordinate-identical across reference releases? | Outcome-blind coordinate qualification complete | `coordinate_identity` |
| [Regional genotype-profile sensitivity](natural_variation/regional_genotype_profile_sensitivity/) | Do regional small-variant profiles define stable descriptive genotype states? | Contract and deterministic derivation complete; result pending | `qtl_vcf_haplotype_sensitivity` |
| [Pangenome-state qualification](natural_variation/pangenome_state_qualification/) | Are genuine regional state calls adequate for state-aware association? | Interfaces qualified; real execution blocked on genuine state calls | `qtl_haplotype_pav`, `qtl_regional_state_prediction_gates` |
| [Held-out-strain prediction](natural_variation/heldout_strain_prediction/) | Can state-aware models predict strains excluded from fitting? | Nested-CV contract tested; real lane blocked on upstream state receipts | `qtl_heldout_prediction` |
| [Phenotype-compendium grouped prediction](natural_variation/phenotype_compendium_grouped_prediction/) | Does genotype/context information predict phenotypes across grouped strain splits? | Source and kernel qualification complete; full result not terminal | `caendr_compendium_*` |
| [Pilot early-to-late prediction](developmental_genetics/pilot_early_to_late_prediction/) | Can early embryo measurements predict later outcomes across unseen perturbations? | Controlled-source pilot complete; initial lineage model not supportive | `developmental_omix` |
| [Processed developmental-data qualification](developmental_genetics/processed_data_qualification/) | Can processed tables be joined, counted, and split without opening sealed outcomes? | Scale-up structure qualified; source-governance gate remains | `developmental_omix_scaleup`, `developmental_raw_archive_identity` |
| [Semantic stage and lineage preparation](developmental_genetics/semantic_stage_and_lineage_preparation/) | Can measurement columns be aligned to a frozen lineage and stage frontier outcome-blind? | 26-cell and 200-cell frontiers plus lineage qualified | `developmental_omix_stage_adapter`, `developmental_readiness` |
| [Gene-disjoint outcome prediction](developmental_genetics/gene_disjoint_outcome_prediction/) | Can models generalize to perturbation genes never used for fitting? | Synthetic contract complete; governed real fitting blocked | `developmental_baselines` |
| [Lineage-topology ablation](developmental_genetics/lineage_topology_ablation/) | Does authentic lineage topology add information beyond capacity and depth controls? | Authentic, edge-free, matched, and rewired controls implemented | `developmental_topology` |
| [WormBase provenance ingestion](context_graph/wormbase_provenance_ingestion/) | Can heterogeneous gene, phenotype, stage, and ontology records retain source context? | Inventory and rights-safe fixtures qualified | `adapters.wormbase_ws298`, `wormbase_ws298_source_qualification` |
| [Graph-grounded reasoner evaluation](biological_reasoning/graph_grounded_reasoner_evaluation/) | Does graph/tool grounding improve valid, evidence-faithful answers over text-only baselines? | Provider-neutral build/replay/score harness tested; no model result | `grounded_reasoner_benchmark` |
| [Prospective RNAi transfer](biological_reasoning/prospective_rnai_transfer/) | Can a preregistered prediction transfer across perturbations or backgrounds? | Blinding and interpretation contracts tested; no real experiment | `prospective_transfer` |

Private operational receipts are not part of this tree. Aggregate public summaries appear
under an experiment's `results/` directory only after their release boundary has been checked.
