# Lecture 3: LQR and Estimation

```bash
uv sync
uv run python -m scripts.run --system double_integrator --controller lqr --estimator perfect
uv run python -m scripts.run --system spacecraft_attitude --estimator kalman
uv run python -m scripts.run --system spacecraft --estimator kalman
uv run python -m scripts.vis_open_loop_and_closed_loop_spectrum
```

- `systems/system.py`: abstract `System`: `dynamics`, `render`, `linearize`, `measure`
- `systems/`: single integrator, double integrator, unicycle (+ parking, not stabilizable), spacecraft translation (Clohessy-Wiltshire), spacecraft attitude
- `systems/spacecraft.py`: coupled translation + attitude (12 states, body-frame thrust + torque) tracking a V-bar docking approach with the docking port pointed at the target
- `util/dare.py`: `dare(A, B, Q, R) -> P, K`
- `controllers/controller.py`, `estimators/estimator.py`: abstract base classes
- `controllers/lqr.py`: LQR about the system reference; `controllers/zero.py`: open loop u = 0
- `estimators/zero.py`: open loop L = 0 (dead reckoning)
- `estimators/kalman.py`: steady-state (extended) Kalman filter via the dual DARE
- `scripts/run.py`: simulates, renders live (`--render_every 0` to disable), saves `traj.npz`, `settings.json`, `plot.png` to `outs/<datetime>_<system>_<controller>_<estimator>_seed<seed>/`
- `scripts/vis_open_loop_and_closed_loop_spectrum.py`: random stabilizable (A, B), plots eig(A) and eig(A - BK) side by side
