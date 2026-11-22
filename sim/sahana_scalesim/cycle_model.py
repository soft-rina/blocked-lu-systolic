"""Cycle formulas supplied by Rina, using physical array dimensions."""
import math


def os_cyc(R, C, M, N, K):
    folds = math.ceil(M / R) * math.ceil(N / C)
    return folds * (R + C + K - 2) - 1


def ws_cyc(R, C, M, N, K):
    folds = math.ceil(K / R) * math.ceil(N / C)
    return folds * (2 * R + C + M - 2) - 1


def lu_updates(n, b):
    for offset in range(0, n, b):
        width = min(b, n - offset)
        remaining = n - offset - width
        if remaining:
            yield remaining, remaining, width


if __name__ == "__main__":
    print("Model predictions only; partial folds require validation.")
    print("n,b,updates,OS_cycles,WS_cycles,OS_over_WS")
    for n in (2048, 4096, 8192):
        for b in (32, 64, 128, 256):
            updates = list(lu_updates(n, b))
            os_total = sum(os_cyc(128, 128, *shape) for shape in updates)
            ws_total = sum(ws_cyc(128, 128, *shape) for shape in updates)
            print(f"{n},{b},{len(updates)},{os_total},{ws_total},{os_total / ws_total:.3f}")
