# Oracle (prior-informed) baselines

Goal: measure how much an active-mapping policy could improve if it knew the scene in advance. Only the *viewpoint
selection* changes; the SplaTAM mapper stays strictly online and never sees the GT geometry, so any change in
coverage, PSNR/SSIM/LPIPS or path length is the value of prior knowledge.

Status legend: `[x]` done, `[ ]` to do. Tick boxes as work lands.

## Background: how ActiveSplat chooses where to go

- The planner (`scripts/nodes/planner_node.py`) builds a Voronoi graph on a top-down free map rendered from the Gaussians.
  Candidates are graph nodes; there is a hierarchical subregion planner and a rotation-selection step on top.
- Each node is scored by *invisibility*: the mapper renders a 360° look-around (3 × 120°×150° views, 1 px = 1°) from the
  node at the agent's camera height and sums `1 - opacity` (`get_global_invisibility`, `get_local_invisibility` in
  `src/mapper/splatam/__init__.py`). The local variant also picks a best rotation target from the largest unmapped cluster.
- So the single swap point for "what is still unseen?" is the look-around image (opacity + depth) those two functions render.

## Corrections to the generic oracle recipe

1. ActiveSplat has no candidate-pose raycasting NBV engine; the swap point is the look-around invisibility above.
2. "Prior knowledge" is separable: (a) which geometry is unseen, (b) which space is navigable, (c) global visit order.
   Tier 2 changes only (a); the tour changes (c) (and uses GT navigability to plan); (b) as a separate toggle stays off.
3. Coverage must be measured on *observable* surface only (reachable from the start floor's navigable island, within
   sensor range and tilt limits), not on every mesh sample: MP3D has several floors, ceilings and wall backs.
4. The offline tour is a heuristic (greedy weighted set cover + TSP), not a true upper bound. Compare it at equal step
   budgets as coverage-vs-steps curves.
5. Poses come from the simulator, so tracking imposes no smoothness constraint on the tour; it only has to be turned into
   discrete actions (0.065 m forward, 10° turns, 15° tilts).

## Metrics

- Exploration: fraction of observable GT surface seen by the sensor, per step (new, `scripts/oracle/coverage.py`).
- Reconstruction: existing `comp_ratio` per step (`eval_actions.py`), final image metrics (`eval_images.py`).
- Headline: area under the coverage-vs-steps curve, steps to 80 % / 90 % coverage, path length, final PSNR/SSIM/LPIPS.
- Extra baselines: random walk (lower bound) and several seeds of the native planner (run-to-run noise).

## Design

- `src/oracle/__init__.py`, `OracleScene`: GT mesh (same frame as `dataset.get_frame()['c2w']`, via
  `scene_mesh_transform`), Open3D `RaycastingScene`, a dense voxel grid (default 5 cm) of *observed* surface, and
  `look_around_view()` returning depth + observed-mask images compatible with the mapper's look-around.
- `planner.oracle` block in the dataset config: `mode` = `native` | `info`, `voxel_size`.
- Tier 2 runs through the same invisibility → node score → hierarchical planner path as native, so candidates, travel cost
  and rotation selection are untouched.
- Tour: precompute observable samples and a visibility matrix, greedy set cover, order by geodesic distance, convert to
  discrete actions, write `actions.txt`, replay through the existing replay mode.

## Work items

### Phase 1: shared oracle pieces
- [x] `OracleScene`: mesh load, raycasting scene, voxel grid, `observe()`, `look_around_view()`
- [x] Unit test: oracle depth vs recorded sensor depth for a few frames of an existing run (validates frames and conventions)
- [x] Observable-surface precompute (`scripts/oracle/precompute_observable.py`), cached per scene under `data/oracle/`
- [x] Coverage scorer for any run (`scripts/oracle/coverage.py`): coverage-vs-step CSV from `actions.txt`

### Phase 2: Tier 2, oracle-info planner
- [x] Config block `planner.oracle` and `config/datasets/*_oracle_info.json`
- [x] Mapper look-around hook (oracle observed-mask + mesh depth replace rendered opacity + depth)
- [x] Visualizer: update observed voxels on every received frame
- [x] Log oracle coverage per step to `oracle_coverage.csv` during the run
- [x] Smoke test on one MP3D scene (300 steps of gZ6f7yhEvPG: runs end to end, diverges from native at step 107; coverage 60.8 % vs native 71.5 % at step 300, a single short run, so not conclusive)
- [x] Runs on the 4 MP3D scenes at high resolution (3 scenes at 1000 steps, see "Results on the other three scenes"; gZ6f7yhEvPG only has the 300-step smoke run, so it still needs a 1000-step run)

### Phase 3: Oracle tour
- [x] Candidate viewpoints on the navmesh (start island): grid x yaw x tilt
- [x] Visibility matrix (observable points x candidates), chunked raycasting
- [x] Greedy lazy set cover to a target coverage (no step budget yet: the tour is cut with `--max_steps`)
- [x] Viewpoint ordering by geodesic distance (nearest neighbour + 2-opt)
- [x] Path + alignment to discrete actions (`actions.txt`)
- [x] Replay through ActiveSplat (mapper replay mode) and evaluate (`mode:=REPLAY` loaded the actions but never used them; implemented in the visualizer's dataset thread)
- [x] Runs on the 4 MP3D scenes (full-length tours; the gZ6f7yhEvPG one is in "First results")

### Phase 4: comparison and baselines
- [ ] Random-walk baseline
- [ ] Native planner repeated with several seeds (noise level)
- [ ] Comparison script/table: coverage curves, AUC, steps-to-X %, path length, image metrics
- [ ] Dashboard tab with the coverage curves

### Later (not started)
- [ ] Tier 3: oracle greedy NBV with free candidate poses on the GT navmesh
- [ ] GT navmesh for navigation (`planner.oracle.navigation: gt_navmesh`)
- [ ] View-count / grazing-angle requirement in the oracle objective (image quality, not just coverage)

## Implementation notes

- Frames: the simulator world frame is the frame of `dataset.get_frame()['c2w']` and of the Gaussians. `scene_mesh_transform`
  gives the evaluation/visualisation frame, which is that frame with y and z negated, so `oracle.load_mesh` undoes it. Camera poses
  for the oracle are OpenCV c2w = `dataset c2w @ OPENCV_TO_OPENGL`. Check: mesh depth vs sensor depth, median error 0.8 cm.
- Range: the oracle uses `dataset.far` (4 m, what the mapper uses), not the simulator sensor's 10 m.
- The oracle needs `data/oracle/<format>_<scene>_observable.npz` (run `scripts/oracle/precompute_observable.py` first).
- Cost: `observe()` about 65 ms per frame and a 360 look-around about 10-40 ms per node on an idle machine; both slower under load.

## First results (gZ6f7yhEvPG only, one run each, so indicative)

| steps | tour | native (1000-step run) |
|---|---|---|
| 100 | 45.1 % | 31.6 % |
| 300 | 79.3 % | 71.5 % |
| 500 | 89.4 % | 86.4 % |
| 700 | 93.5 % | 87.5 % |
| 964 | 98.7 % | 89.2 % |

Observable-surface coverage by the sensor; the tour is 964 actions and 22.4 m (native: 16.6 m in 963 steps). Image metrics on the
trajectory views: PSNR 19.6 vs 17.4, SSIM 0.749 vs 0.725, LPIPS 0.267 vs 0.257, mapped-pixel PSNR 24.0 vs 24.9: the tour
covers more of the scene, while the quality of what is mapped is similar. `eval_actions` completion ratio 97.7 % for the tour.

## Results on the other three scenes (one run each, so indicative)

Observable-surface coverage (%) of the native planner, the oracle-info planner (1000 steps) and the oracle tour replayed in full
(tour lengths: 1248, 1901, 1837 actions, so only the first 1000 are comparable with the others).

| scene | method | 100 | 300 | 500 | 1000 | final | path (m) |
|---|---|---|---|---|---|---|---|
| GdvgFV5R1Z5 | native | 53.6 | 70.1 | 78.2 | 86.7 | 86.7 | 68.4 |
| | oracle-info | 53.6 | 69.9 | 78.9 | 92.9 | 92.9 | 28.8 |
| | tour | 51.1 | 66.7 | 72.6 | 84.7 | 98.6 (1248) | 29.4 |
| pLe4wQe7qrG | native | 23.9 | 30.9 | 42.1 | 83.1 | 83.1 | 14.4 |
| | oracle-info | 23.1 | 47.1 | 52.3 | 85.4 | 85.4 | 21.0 |
| | tour | 23.6 | 39.2 | 51.3 | 71.0 | 98.7 (1901) | 49.3 |
| YmJkqBEsHnH | native | 21.4 | 39.7 | 52.1 | 62.5 | 62.5 | 22.8 |
| | oracle-info | 21.4 | 32.6 | 57.6 | 81.7 | 81.7 | 21.5 |
| | tour | 26.1 | 47.2 | 59.8 | 64.9 | 99.2 (1837) | 49.7 |

Image metrics on the trajectory views (PSNR / SSIM / LPIPS; tours are scored over their full length, the others over 1000 steps):

| scene | native | oracle-info | tour |
|---|---|---|---|
| GdvgFV5R1Z5 | 24.6 / 0.794 / 0.203 | 24.2 / 0.780 / 0.220 | 23.0 / 0.734 / 0.287 |
| pLe4wQe7qrG | 26.0 / 0.785 / 0.252 | 25.6 / 0.769 / 0.273 | 24.4 / 0.741 / 0.312 |
| YmJkqBEsHnH | 15.3 / 0.644 / 0.326 | 19.7 / 0.765 / 0.244 | 16.1 / 0.638 / 0.363 |

- Oracle-info beats native at 1000 steps on all three scenes (+6.2, +2.3, +19.2 points) and clearly improves image quality on
  YmJkqBEsHnH, where native leaves much unseen. Where native already covers most of the scene its image metrics are slightly better.
- The tour only pays off when run to the end (about 99 % coverage), which takes 1248 to 1901 actions. Cut at 1000 steps it
  is level with or behind native, and its image metrics are lower than native on two of the three scenes: coverage is not
  image quality, and it takes the tour longer paths (up to 50 m) and many turns.
- The new native rerun of gZ6f7yhEvPG reaches 31.6 / 58.9 / 86.9 / 94.3 % at 100 / 300 / 500 / 1000 steps (the first results
  above used an older native run, which was weaker after step 500). Its trajectory metrics: PSNR 17.6, SSIM 0.718, LPIPS 0.271.

## Tour planner notes

- 15 s end to end (was over 40 min unfinished): k-d tree over the target subsample, O(1) 2-opt moves, geodesics per distinct position.
- Camera pitches -30/0/+30 deg are candidates (without them the cover stalls at 77.5 %); views at one position become one sweep.
- Execution uses habitat's greedy geodesic follower; 60 % of the actions are still turns, so heading-aware ordering is a next step.

## Risks and notes

- The look-around runs in the mapper inside the visualizer process, so the oracle scene is loaded there once per run.
- The oracle cost per node: one raycast of 3 x 120 x 150 rays (about 54k) plus a voxel lookup, aiming for tens of ms.
- Each run takes 15-60 min on the shared GPUs: study = 4 scenes x several tiers x seeds; schedule within the 2-GPU rule.
- The sim's depth sensor and the mesh may differ slightly (mesh holes, sensor noise off); validate with the Phase 1 test.
