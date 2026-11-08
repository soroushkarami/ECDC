# verify_mac_solver.py
import sys; sys.path.insert(0, '.')
from net_metrics import (_solve_bianchi_fixed_point,
                          _get_collision_probability,
                          _find_n_max_bianchi,
                          _get_collision_probability_broadcast)

REF = {10: 0.388301, 20: 0.476909}
print("  n | solver p    | R2 ref p    | |Δ|       | residual  | converged")
print("-" * 74)
for n in [2, 3, 5, 10, 15, 20, 22, 24, 25, 26, 30, 50]:
    p, conv, res = _solve_bianchi_fixed_point(n)
    ref = REF.get(n)
    print(f"{n:3d} | {p if p is not None else 'NONE':>11} | "
          f"{(f'{ref:.6f}' if ref else '—'):>11} | "
          f"{(f'{abs(p-ref):.2e}' if ref and p else '—'):>9} | "
          f"{res:.1e} | {conv}")

n_max = _find_n_max_bianchi()
print(f"\nBianchi fixed point exists for n ≤ {n_max}")

print("\nBroadcast p at n = 2, 10, 25, 50, 60:")
for n in [2, 10, 25, 50, 60]:
    print(f"  n={n:3d}  p_bcast = {_get_collision_probability_broadcast(n):.4f}")

print("\nOperational _get_collision_probability (load_factor=1):")
for n in [10, 20, 30, 60]:
    print(f"  n={n:3d}  p = {_get_collision_probability(n):.6f}")