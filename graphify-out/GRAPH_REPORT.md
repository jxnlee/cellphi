# Graph Report - cellphi  (2026-11-12)

## Corpus Check
- Corpus is ~15,980 words - fits in a single context window. You may not need a graph.

## Summary
- 275 nodes · 635 edges · 17 communities (12 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.88)
- Token cost: 57,013 input · 0 output

## Community Hubs (Navigation)
- CLI and Package Entry
- Size Estimate and Fallback
- Mask Measurement and CSV
- Annotation Drawing
- GitHub Actions Workflow
- CLI Tests and Diameters
- Reference Data Tooling
- Image Folder Search
- Pipeline Design Docs
- Calibration and Testing Docs
- Cellpose Segmentation Wrapper
- Pixel Calibration
- Pre-push Check Script
- Coding Conventions
- Robustness Notes
- Package Metadata

## God Nodes (most connected - your core abstractions)
1. `measure_masks()` - 30 edges
2. `main()` - 23 edges
3. `process_image()` - 20 edges
4. `estimate_diameter_px()` - 17 edges
5. `find_images()` - 17 edges
6. `annotate()` - 15 edges
7. `Calibration` - 14 edges
8. `Organoid` - 14 edges
9. `add_missed_dark_blobs()` - 12 edges
10. `diameter_line()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `summary()` --calls--> `main()`  [EXTRACTED]
  tests/test_pipeline_real.py → cellphi/cli.py
- `test_annotate_draws_legend_and_line()` --calls--> `annotate()`  [EXTRACTED]
  tests/test_measure.py → cellphi/draw.py
- `test_feret_single_pixel()` --calls--> `max_feret()`  [EXTRACTED]
  tests/test_measure.py → cellphi/measure.py
- `test_diameter_line_length_equals_reported_diameter()` --calls--> `diameter_line()`  [EXTRACTED]
  tests/test_measure.py → cellphi/measure.py
- `test_min_feret_line_spans_the_narrow_axis()` --calls--> `diameter_line()`  [EXTRACTED]
  tests/test_measure.py → cellphi/measure.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Segmentation and cleanup flow (size hint, Cellpose, dark fallback, fragment merge, edge flag)** — agents_size_hint_estimate, agents_cellpose_cpsam, agents_dark_blob_fallback, agents_overlap_fragments, agents_edge_organoid_flagging [EXTRACTED 1.00]
- **Validation against manual ImageJ measurements** — agents_reference_manual_measurements, agents_default_calibration, agents_equivalent_diameter_default, readme_accuracy [INFERRED 0.85]
- **Testing and CI infrastructure** — agents_github_actions_ci, agents_fast_vs_slow_tests, agents_check_before_push, requirements_ci_requirements_ci, agents_ci_no_model_imports [EXTRACTED 1.00]

## Communities (17 total, 5 thin omitted)

### Community 0 - "CLI and Package Entry"
Cohesion: 0.11
Nodes (10): build_parser(), _size(), Organoid, _for_cellpose(), ImageResult, process_image(), _stats(), summary_row() (+2 more)

### Community 1 - "Size Estimate and Fallback"
Cohesion: 0.13
Nodes (18): add_missed_dark_blobs(), dark_foreground(), estimate_diameter_px(), to_gray8(), read_image(), brightfield(), grid_disks(), test_dark_foreground_polarity_from_masks() (+10 more)

### Community 2 - "Mask Measurement and CSV"
Cohesion: 0.19
Nodes (22): _diameter(), measure_masks(), number_reading_order(), organoid_rows(), disk_mask(), test_annotate_draws_legend_and_line(), test_border_margin_catches_masks_stopping_short_of_edge(), test_default_calibration_matches_manual_measurements() (+14 more)

### Community 3 - "Annotation Drawing"
Cohesion: 0.17
Nodes (8): annotate(), _put_legend(), _put_text_block(), to_bgr8(), caliper_widths(), diameter_line(), max_feret(), _measure_region()

### Community 4 - "GitHub Actions Workflow"
Cohesion: 0.09
Nodes (22): os, python, jobs, test, include, name, on, pull_request (+14 more)

### Community 5 - "CLI Tests and Diameters"
Cohesion: 0.15
Nodes (10): main(), _Device, fake_cellpose(), FakeSegmenter, read_csv(), test_cli_default_input_folder(), test_cli_end_to_end(), test_cli_options() (+2 more)

### Community 6 - "Reference Data Tooling"
Cohesion: 0.13
Nodes (6): summary(), test_counts_and_sizes_match_reviewed_run(), extract_well(), find_measured_image(), main(), rebuild_lines()

### Community 7 - "Image Folder Search"
Cohesion: 0.19
Nodes (14): find_images(), add(), ImageInput, _is_cellphi_output(), names(), test_files_and_duplicates(), test_missing_input(), test_non_recursive() (+6 more)

### Community 8 - "Pipeline Design Docs"
Cohesion: 0.13
Nodes (13): AGENTS.md (agent/contributor guide), Cellphi organoid measurement pipeline, Cellpose v4 cpsam model, Unsupervised coverage/spill check (Otsu dark pixels), Goal: organoid-specific fine-tuned Cellpose model, input/ dataset (Cortical and Midbrain organoids, 2048x1536 JPEG), Known limitations (hidden organoids, brightfield assumption), CLAUDE.md (imports AGENTS.md) (+5 more)

### Community 9 - "Calibration and Testing Docs"
Cohesion: 0.13
Nodes (12): tools/check_before_push.sh and .githooks/pre-push, Fast (CI) vs slow (Cellpose) test split, GitHub Actions CI (.github/workflows/tests.yml), reference/ manual ImageJ measurements (Plate 1 CM_CM, 6 wells), Target environments (WSL2 CUDA, macOS, Colab), Accuracy vs manual ImageJ (284/285 found, median ratio 1.00), Calibration procedure (--scale-bar, --reference-size, --units-per-px), Diameter definitions (equivalent, mean_feret, max_feret, min_feret) (+4 more)

### Community 10 - "Cellpose Segmentation Wrapper"
Cohesion: 0.13
Nodes (4): pick_device(), Segmenter, comparisons(), test_diameters_match_manual_measurements()

### Community 11 - "Pixel Calibration"
Cohesion: 0.27
Nodes (4): Calibration, calibration_from_args(), test_calibration_scales_with_resolution(), test_calibration_warns_on_aspect_mismatch()

## Knowledge Gaps
- **10 isolated node(s):** `cellphi`, `CLAUDE.md (imports AGENTS.md)`, `Robustness to resolution and contrast`, `tools/check_before_push.sh and .githooks/pre-push`, `Known limitations (hidden organoids, brightfield assumption)` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 93 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `CLI Tests and Diameters` to `CLI and Package Entry`, `Mask Measurement and CSV`, `Annotation Drawing`, `Reference Data Tooling`, `Image Folder Search`, `Cellpose Segmentation Wrapper`, `Pixel Calibration`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `cellphi`, `CLAUDE.md (imports AGENTS.md)`, `Robustness to resolution and contrast` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CLI and Package Entry` be split into smaller, more focused modules?**
  _Cohesion score 0.11260504201680673 - nodes in this community are weakly interconnected._
- **Why does `process_image()` connect `CLI and Package Entry` to `Size Estimate and Fallback`, `Mask Measurement and CSV`, `CLI Tests and Diameters`, `Image Folder Search`, `Cellpose Segmentation Wrapper`, `Pixel Calibration`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `Size Estimate and Fallback` be split into smaller, more focused modules?**
  _Cohesion score 0.13068181818181818 - nodes in this community are weakly interconnected._
- **Why does `find_images()` connect `Image Folder Search` to `CLI and Package Entry`, `CLI Tests and Diameters`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `GitHub Actions Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._