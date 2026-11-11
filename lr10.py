"""
Лабораторная работа № 10. Вариант 17.
Графы: решётка k×k, k = 20, 40, 80 (n = 400, 1600, 6400).
Веса 1…25. Дейкстра: массив Θ(n²) и куча O(m log n).
Критический путь: 12 работ.
Доп. К3: время обеих реализаций от плотности графа.
"""
from __future__ import annotations

import heapq
import json
import random
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT = Path("/home/workdir/artifacts/lr10")
CODE = OUT / "code"
DIAG = OUT / "diagrams"
CODE.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

INF = float("inf")


def dijkstra_array(adj, s):
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    used = [False] * n
    dist[s] = 0
    for _ in range(n):
        v, best = -1, INF
        for i in range(n):
            if not used[i] and dist[i] < best:
                v, best = i, dist[i]
        if v == -1:
            break
        used[v] = True
        for u, w in adj[v]:
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                parent[u] = v
    return dist, parent


def dijkstra_heap(adj, s):
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    dist[s] = 0
    heap = [(0, s)]
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v]:
            continue
        for u, w in adj[v]:
            nd = d + w
            if nd < dist[u]:
                dist[u] = nd
                parent[u] = v
                heapq.heappush(heap, (nd, u))
    return dist, parent


def restore_path(parent, target):
    path = []
    while target != -1:
        path.append(target)
        target = parent[target]
    return path[::-1]


def generate_grid(k, max_weight=25, seed=42):
    """Ориентированная решётка k×k: рёбра вправо и вниз (+ случайные вверх/влево)."""
    rng = random.Random(seed + k)
    n = k * k
    adj = [[] for _ in range(n)]

    def idx(r, c):
        return r * k + c

    for r in range(k):
        for c in range(k):
            v = idx(r, c)
            if c + 1 < k:
                adj[v].append((idx(r, c + 1), rng.randint(1, max_weight)))
            if r + 1 < k:
                adj[v].append((idx(r + 1, c), rng.randint(1, max_weight)))
            # немного обратных для разнообразия путей
            if c > 0 and rng.random() < 0.15:
                adj[v].append((idx(r, c - 1), rng.randint(1, max_weight)))
            if r > 0 and rng.random() < 0.15:
                adj[v].append((idx(r - 1, c), rng.randint(1, max_weight)))
    m = sum(len(lst) for lst in adj)
    return adj, m


def generate_random(n, density, max_weight=25, seed=7):
    """Случайный орграф с заданной плотностью (доля от n*(n-1))."""
    rng = random.Random(seed + n)
    adj = [[] for _ in range(n)]
    # остов для связности из 0
    for i in range(1, n):
        j = rng.randrange(i)
        adj[j].append((i, rng.randint(1, max_weight)))
    max_edges = n * (n - 1)
    target = max(n - 1, int(max_edges * density))
    added = n - 1
    tries = 0
    while added < target and tries < target * 20:
        u, v = rng.randrange(n), rng.randrange(n)
        tries += 1
        if u != v:
            adj[u].append((v, rng.randint(1, max_weight)))
            added += 1
    m = sum(len(lst) for lst in adj)
    return adj, m


# ---------- Критический путь ----------
def topological_order(tasks):
    indeg = {k: len(v[1]) for k, v in tasks.items()}
    succ = {k: [] for k in tasks}
    for k, (_, preds) in tasks.items():
        for p in preds:
            succ[p].append(k)
    queue = deque(sorted(k for k, d in indeg.items() if d == 0))
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(u)
    if len(order) != len(tasks):
        raise ValueError("Сетевой график содержит цикл")
    return order, succ


def critical_path(tasks):
    order, succ = topological_order(tasks)
    ES, EF = {}, {}
    for v in order:
        duration, preds = tasks[v]
        ES[v] = max((EF[p] for p in preds), default=0)
        EF[v] = ES[v] + duration
    T = max(EF.values())
    LS, LF = {}, {}
    for v in reversed(order):
        duration = tasks[v][0]
        LF[v] = min((LS[u] for u in succ[v]), default=T)
        LS[v] = LF[v] - duration
    reserve = {v: LS[v] - ES[v] for v in tasks}
    critical = [v for v in order if reserve[v] == 0]
    return T, ES, EF, LS, LF, reserve, critical, order


def main():
    # --- 1. Малый граф ---
    print("=== Малый граф (проверка) ===")
    small_adj = [
        [(1, 4), (2, 2)],
        [(2, 1), (3, 5)],
        [(3, 8), (4, 10)],
        [(4, 2)],
        [],
    ]
    d_arr, p_arr = dijkstra_array(small_adj, 0)
    d_heap, p_heap = dijkstra_heap(small_adj, 0)
    assert d_arr == d_heap
    print(f"  dist from 0: {d_arr}")
    print(f"  path to 4: {restore_path(p_heap, 4)}")

    # --- 2. Решётки ---
    print("\n=== Решётки k×k ===")
    grid_rows = []
    print(f"{'k':>4} {'n':>6} {'m':>8} {'array,s':>10} {'heap,s':>10} {'ratio':>8} {'dist_end':>10}")
    for k in [20, 40, 80]:
        adj, m = generate_grid(k)
        n = k * k
        t0 = time.perf_counter()
        d1, _ = dijkstra_array(adj, 0)
        t_a = time.perf_counter() - t0
        t0 = time.perf_counter()
        d2, p2 = dijkstra_heap(adj, 0)
        t_h = time.perf_counter() - t0
        assert d1 == d2
        end = n - 1
        row = {
            "k": k, "n": n, "m": m,
            "t_array": t_a, "t_heap": t_h,
            "ratio": t_a / t_h if t_h > 0 else 0,
            "dist_end": d2[end] if d2[end] < INF else None,
            "path_len": len(restore_path(p2, end)),
        }
        grid_rows.append(row)
        print(f"{k:>4} {n:>6} {m:>8} {t_a:>10.4f} {t_h:>10.4f} {row['ratio']:>8.1f} {row['dist_end']:>10}")

    # --- 3. К3: плотность ---
    print("\n=== К3: время vs плотность (n=400) ===")
    density_rows = []
    n_fix = 400
    for dens in [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]:
        adj, m = generate_random(n_fix, dens)
        t0 = time.perf_counter()
        dijkstra_array(adj, 0)
        t_a = time.perf_counter() - t0
        t0 = time.perf_counter()
        dijkstra_heap(adj, 0)
        t_h = time.perf_counter() - t0
        density_rows.append({
            "density": dens, "m": m,
            "t_array": t_a, "t_heap": t_h,
            "ratio": t_a / t_h if t_h > 0 else 0,
        })
        print(f"  dens={dens:.0%} m={m:>7} array={t_a:.4f}s heap={t_h:.4f}s ratio={t_a/t_h:.1f}")

    # --- 4. Критический путь (12 работ) ---
    tasks = {
        "A": (4, []),
        "B": (6, ["A"]),
        "C": (5, ["A"]),
        "D": (3, ["B"]),
        "E": (7, ["B", "C"]),
        "F": (4, ["C"]),
        "G": (5, ["D", "E"]),
        "H": (3, ["E", "F"]),
        "I": (6, ["G"]),
        "J": (2, ["G", "H"]),
        "K": (4, ["H"]),
        "L": (3, ["I", "J", "K"]),
    }
    T, ES, EF, LS, LF, R, crit, order = critical_path(tasks)
    print(f"\n=== Критический путь: T={T} дн. ===")
    print(f"{'Раб':>4} {'t':>3} {'ES':>4} {'EF':>4} {'LS':>4} {'LF':>4} {'R':>4} {'крит':>5}")
    cpm_table = []
    for v in order:
        cpm_table.append({
            "code": v, "t": tasks[v][0],
            "ES": ES[v], "EF": EF[v], "LS": LS[v], "LF": LF[v], "R": R[v],
            "critical": R[v] == 0,
        })
        print(f"{v:>4} {tasks[v][0]:>3} {ES[v]:>4} {EF[v]:>4} {LS[v]:>4} {LF[v]:>4} {R[v]:>4} "
              f"{'да' if R[v]==0 else '':>5}")
    print("Критический путь:", " → ".join(crit))

    out = {
        "grid": grid_rows,
        "density": density_rows,
        "cpm": {"T": T, "critical": crit, "table": cpm_table},
        "small": {"dist": d_arr, "path_to_4": restore_path(p_heap, 4)},
    }
    with open(CODE / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("results.json saved")

    # Graphs
    font_path = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Regular.ttf"
    font_bold = "/usr/share/fonts/SlidesCarnival/google/Tinos/Tinos-Bold.ttf"
    prop = fm.FontProperties(fname=font_path) if Path(font_path).exists() else None
    prop_b = fm.FontProperties(fname=font_bold) if Path(font_bold).exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ks = [r["k"] for r in grid_rows]
    axes[0].plot(ks, [r["t_array"] for r in grid_rows], "o-", label="Массив Θ(n²)", color="#c62828")
    axes[0].plot(ks, [r["t_heap"] for r in grid_rows], "s-", label="Куча O(m log n)", color="#1565c0")
    axes[0].set_xlabel("k (решётка k×k)", fontproperties=prop)
    axes[0].set_ylabel("Время, с", fontproperties=prop)
    axes[0].set_title("Дейкстра на решётках", fontproperties=prop_b)
    axes[0].legend(prop=prop, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_yscale("log")

    dens = [r["density"] * 100 for r in density_rows]
    axes[1].plot(dens, [r["t_array"] for r in density_rows], "o-", label="Массив", color="#c62828")
    axes[1].plot(dens, [r["t_heap"] for r in density_rows], "s-", label="Куча", color="#1565c0")
    axes[1].set_xlabel("Плотность, %", fontproperties=prop)
    axes[1].set_ylabel("Время, с (n=400)", fontproperties=prop)
    axes[1].set_title("К3: время vs плотность", fontproperties=prop_b)
    axes[1].legend(prop=prop, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            if prop:
                label.set_fontproperties(prop)
            label.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(DIAG / "dijkstra_bench.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("dijkstra_bench.png saved")


if __name__ == "__main__":
    main()
