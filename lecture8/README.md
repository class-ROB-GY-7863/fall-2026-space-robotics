# Lecture 8: Constrained Minimization

Edit `solvers.py` only. The functions are the algorithms from the lecture slides.

```bash
cd lecture8
uv run python -m scripts.a_unconstrained   # Part A: gradient descent, contraction rate c
uv run python -m scripts.b_equality        # Part B: equality constraints, primal-dual with and without eta
uv run python -m scripts.c_thrusters       # Part C: thruster allocation, equality and inequality constraints
```

Each script saves `plot.png` to `outs/<datetime>_<part>/`.

- `problems.py`: test problems (do not edit)
- `solvers.py`: the functions you write

## Deliverable (hand in at end of class)

One page with your name. For each part: the plot and one or two sentences.

- A. Does the dashed bound `c^k` hold? What happens when `alpha > 2/M`?
- B. Plain primal-dual (`eta = 0`): is `rho(A) < 1`? Is `||A|| < 1`? Does it converge? Why is this not a contradiction?
  With `eta > 0`: does the bound `c^k` hold? How far is the answer from the true solution? Why?
- C. How does `eta` change the error and the constraint violation? The run uses `alpha = 0.1`, which is larger than
  the bound `2 mu / ell^2`. Does it still converge? What does this tell you about the bound?
