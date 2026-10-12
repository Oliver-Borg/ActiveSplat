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
- [ ] `OracleScene`: mesh load, raycasting scene, voxel grid, `observe()`, `look_around_view()`
- [ ] Unit test: oracle depth vs recorded sensor depth for a few frames of an existing run (validates frames and conventions)
- [ ] Observable-surface precompute (`scripts/oracle/precompute_observable.py`), cached per scene under `data/oracle/`
- [ ] Coverage scorer for any run (`scripts/oracle/coverage.py`): coverage-vs-step CSV from `actions.txt`

### Phase 2: Tier 2, oracle-info planner
- [ ] Config block `planner.oracle` and `config/datasets/*_oracle_info.json`
- [ ] Mapper look-around hook (oracle observed-mask + mesh depth replace rendered opacity + depth)
- [ ] Visualizer: update observed voxels on every received frame
- [ ] Log oracle coverage per step to `oracle_coverage.csv` during the run
- [ ] Smoke test on one MP3D scene (check it explores, compare steps to native)
- [ ] Runs on the 4 MP3D scenes at high resolution

### Phase 3: Oracle tour
- [ ] Candidate viewpoints on the navmesh (start island): grid x yaw x tilt
- [ ] Visibility matrix (observable points x candidates), chunked raycasting
- [ ] Greedy weighted set cover with a step budget
- [ ] Viewpoint ordering by geodesic distance (nearest neighbour + 2-opt)
- [ ] Path + alignment to discrete actions (`actions.txt`)
- [ ] Replay through ActiveSplat (mapper replay mode) and evaluate
- [ ] Runs on the 4 MP3D scenes

### Phase 4: comparison and baselines
- [ ] Random-walk baseline
- [ ] Native planner repeated with several seeds (noise level)
- [ ] Comparison script/table: coverage curves, AUC, steps-to-X %, path length, image metrics
- [ ] Dashboard tab with the coverage curves

### Later (not started)
- [ ] Tier 3: oracle greedy NBV with free candidate poses on the GT navmesh
- [ ] GT navmesh for navigation (`planner.oracle.navigation: gt_navmesh`)
- [ ] View-count / grazing-angle requirement in the oracle objective (image quality, not just coverage)

## Risks and notes

- The look-around runs in the mapper inside the visualizer process, so the oracle scene is loaded there once per run.
- The oracle cost per node: one raycast of 3 x 120 x 150 rays (about 54k) plus a voxel lookup, aiming for tens of ms.
- Each run takes 15-60 min on the shared GPUs: study = 4 scenes x several tiers x seeds; schedule within the 2-GPU rule.
- The sim's depth sensor and the mesh may differ slightly (mesh holes, sensor noise off); validate with the Phase 1 test.
