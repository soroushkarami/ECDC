import math

def bianchi_fixed_point(N, W, m):
    tau = 2.0 / (W + 1)
    p = 0.0
    for _ in range(200):
        p_new = 1.0 - (1.0 - tau) ** (N - 1)
        num = 2.0 * (1.0 - 2.0 * p_new)
        den = (1.0 - 2.0 * p_new) * (W + 1) + p_new * W * (1.0 - (2.0 * p_new) ** m)
        if den == 0:
            break
        tau_new = num / den
        if abs(tau_new - tau) < 1e-12 and abs(p_new - p) < 1e-12:
            tau, p = tau_new, p_new
            break
        tau, p = tau_new, p_new
    return tau, p

def bianchi_throughput(N, tau):
    # FHSS parameters from Bianchi Table II
    sigma   = 50e-6                  # slot time in seconds
    Ts      = 8982e-6                # successful transmission duration
    Tc      = 8713e-6                # collision duration
    EP_bits = 8184                   # payload bits
    P_tr = 1.0 - (1.0 - tau) ** N
    if P_tr <= 0:
        return 0.0
    P_s = N * tau * (1.0 - tau) ** (N - 1) / P_tr
    num = P_tr * P_s * EP_bits
    den = (1.0 - P_tr) * sigma + P_tr * P_s * Ts + P_tr * (1.0 - P_s) * Tc
    return num / den / 1e6  # normalize to channel bit rate

print(f"{'N':>4} {'p':>10} {'tau':>10} {'S':>10}")
for N in [2, 3, 5, 10, 20, 30, 50]:
    tau, p = bianchi_fixed_point(N, W=32, m=3)
    S = bianchi_throughput(N, tau)
    print(f"{N:>4} {p:>10.4f} {tau:>10.4f} {S:>10.4f}")