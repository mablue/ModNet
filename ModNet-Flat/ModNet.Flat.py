"""
ModNet Benchmark Suite + Spring Graph (v2)
============================================
- Fixed: multi-output support (full_adder, adder2bit, comparator, sort4)
- Fixed: per-problem seed independence
- Spring graph visualization with networkx
- 16 benchmark problems

Install:
  pip install networkx matplotlib
"""
import os, json, csv, time, random, math
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("Warning: networkx or matplotlib not installed.")
    print("Run: pip install networkx matplotlib")

SEED = 42

OUT = Path('results_bench_v2')
OUT.mkdir(exist_ok=True)

PALETTE_HIDDEN = '#2E86AB'
PALETTE_OUTPUT = '#C73E1D'
PALETTE_INPUT  = '#0a8a3a'


# ============================================================
#                     ModNet
# ============================================================
class ModNet:
    def __init__(self, n_in, n_out, total):
        self.n_in = n_in
        self.n_out = n_out
        self.total = total
        self.out_idx = list(range(n_out))
        self._alloc()
        self._init()

    def _alloc(self):
        N = self.total
        self.W = np.zeros((N, N), dtype=np.float32)
        self.Win = np.zeros((N, self.n_in), dtype=np.float32)
        self.bias = np.zeros(N, dtype=np.float32)

    def _init(self):
        """Output nodes cannot connect directly to inputs."""
        for i in range(self.total):
            is_output = i in self.out_idx
            for _ in range(random.randint(1, 3)):
                if (not is_output) and random.random() < 0.3:
                    self.Win[i, random.randrange(self.n_in)] += random.gauss(0, 1)
                else:
                    self.W[i, random.randrange(self.total)] += random.gauss(0, 1)
            self.bias[i] = random.gauss(0, 1)

    def forward_batch(self, X, steps):
        B = X.shape[0]
        state = np.zeros((B, self.total), dtype=np.float32)
        const = X @ self.Win.T + self.bias
        Wt = self.W.T
        for _ in range(steps):
            state = ((state @ Wt + const) >= 0).astype(np.float32)
        return state[:, self.out_idx]

    def predict_values(self, X, steps, POWERS, DENOM):
        """Scalar decoding for regression & single-output boolean."""
        out = self.forward_batch(X, steps)
        return (out @ POWERS) / DENOM

    def predict_bits(self, X, steps):
        """Raw output bits — for multi-output boolean."""
        return self.forward_batch(X, steps)[:, :self.n_out]

    def activity(self, X, steps):
        B = X.shape[0]
        state = np.zeros((B, self.total), dtype=np.float32)
        const = X @ self.Win.T + self.bias
        Wt = self.W.T
        total_act = np.zeros(self.total, dtype=np.float64)
        for _ in range(steps):
            state = ((state @ Wt + const) >= 0).astype(np.float32)
            total_act += state.mean(axis=0)
        return total_act / steps

    def fitness(self, X, Y, steps, POWERS, DENOM, kind):
        out = self.forward_batch(X, steps)
        if Y.ndim == 2:
            diffs = (out[:, :self.n_out] - Y) ** 2
            return float(-np.mean(diffs))
        else:
            preds = (out @ POWERS) / DENOM
            return float(-np.mean((preds - Y) ** 2))

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.n_out = self.n_out
        n.total = self.total
        n.out_idx = list(self.out_idx)
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        return n

    def mutate(self, rate=2, sigma=0.5):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            is_output = i in new.out_idx
            r = random.random()
            if r < 0.40:
                new.W[i, random.randrange(new.total)] += random.gauss(0, sigma)
            elif r < 0.60:
                new.bias[i] += random.gauss(0, sigma)
            elif r < 0.80:
                if not is_output:
                    new.Win[i, random.randrange(new.n_in)] += random.gauss(0, sigma)
                else:
                    new.W[i, random.randrange(new.total)] += random.gauss(0, sigma)
            elif r < 0.92:
                nz = np.nonzero(new.W[i])[0]
                if len(nz) > 1:
                    new.W[i, int(random.choice(nz))] = 0
            else:
                new.W[i, random.randrange(new.total)] = random.gauss(0, 1)
        return new

    def birth(self, max_total=60):
        if self.total >= max_total:
            return None
        new = self.clone()
        N = self.total
        new_W = np.zeros((N + 1, N + 1), dtype=np.float32)
        new_W[:N, :N] = new.W
        new.W = new_W
        new_Win = np.zeros((N + 1, self.n_in), dtype=np.float32)
        new_Win[:N, :] = new.Win
        new.Win = new_Win
        new.bias = np.append(new.bias, random.gauss(0, 1))
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.3:
                new.Win[N, random.randrange(self.n_in)] += random.gauss(0, 1)
            else:
                new.W[N, random.randrange(N)] += random.gauss(0, 1)
        new.total = N + 1
        return new

    def crossover(self, other):
        if self.total != other.total:
            return None
        new = self.clone()
        for i in range(new.total):
            if random.random() < 0.5:
                new.W[i] = other.W[i].copy()
                new.Win[i] = other.Win[i].copy()
                new.bias[i] = other.bias[i]
        return new

    def _compute_reachability(self):
        total = self.total
        out_set = set(self.out_idx)
        fwd = np.zeros(total, dtype=bool)
        direct_in = (np.abs(self.Win).sum(axis=1) > 0)
        fwd |= direct_in
        for _ in range(total):
            new_fwd = fwd.copy()
            for i in range(total):
                if fwd[i]:
                    out_edges = np.nonzero(self.W[:, i])[0]
                    new_fwd[out_edges] = True
            if np.array_equal(new_fwd, fwd):
                break
            fwd = new_fwd
        bwd = np.zeros(total, dtype=bool)
        for i in out_set:
            bwd[i] = True
        for _ in range(total):
            new_bwd = bwd.copy()
            for i in range(total):
                if bwd[i]:
                    in_edges = np.nonzero(self.W[i])[0]
                    new_bwd[in_edges] = True
            if np.array_equal(new_bwd, bwd):
                break
            bwd = new_bwd
        return fwd & bwd

    def _prune_inplace(self, X, steps, w_thresh=0.04,
                       act_lo=0.02, act_hi=0.98, min_hidden=2):
        act = self.activity(X, steps)
        self.W[np.abs(self.W) < w_thresh] = 0
        self.Win[np.abs(self.Win) < w_thresh] = 0
        alive = np.ones(self.total, dtype=bool)
        dead = (act < act_lo) | (act > act_hi)
        alive &= ~dead
        in_from_W = np.abs(self.W).sum(axis=0)
        in_from_in = np.abs(self.Win).sum(axis=1)
        out_to = np.abs(self.W).sum(axis=1)
        isolated = (in_from_W + in_from_in == 0) & (out_to == 0)
        alive &= ~isolated
        reachable = self._compute_reachability()
        alive &= reachable
        for i in self.out_idx:
            alive[i] = True
        hidden_alive = [i for i in range(self.total)
                        if alive[i] and i not in self.out_idx]
        if len(hidden_alive) < min_hidden:
            hidden_all = [i for i in range(self.total) if i not in self.out_idx]
            sorted_h = sorted(hidden_all, key=lambda i: -abs(act[i] - 0.5))
            for i in sorted_h[:min_hidden]:
                alive[i] = True
        keep = np.array([i for i in range(self.total) if alive[i]], dtype=np.int32)
        old_total = self.total
        new_total = len(keep)
        if new_total == old_total:
            return 0
        old_to_new = -np.ones(old_total, dtype=np.int32)
        for i_new, i_old in enumerate(keep):
            old_to_new[i_old] = i_new
        new_W = np.zeros((new_total, new_total), dtype=np.float32)
        new_Win = np.zeros((new_total, self.n_in), dtype=np.float32)
        new_bias = np.zeros(new_total, dtype=np.float32)
        for i_new, i_old in enumerate(keep):
            new_bias[i_new] = self.bias[i_old]
            new_Win[i_new] = self.Win[i_old]
            mask = old_to_new >= 0
            new_W[i_new, old_to_new[mask]] = self.W[i_old, mask]
        self.W = new_W
        self.Win = new_Win
        self.bias = new_bias
        self.total = new_total
        self.out_idx = list(range(self.n_out))
        return old_total - new_total

    def try_prune(self, X, Y, steps, POWERS, DENOM, kind, old_fit=None):
        if old_fit is None:
            old_fit = self.fitness(X, Y, steps, POWERS, DENOM, kind)
        cand = self.clone()
        removed = cand._prune_inplace(X, steps)
        if removed == 0:
            return None
        nf = cand.fitness(X, Y, steps, POWERS, DENOM, kind)
        if nf >= old_fit - 0.001:
            return cand, removed, nf
        return None

    def stats(self):
        Wnz = np.nonzero(self.W)
        rec = sum(1 for i, j in zip(Wnz[0], Wnz[1]) if i == j)
        return rec, int(self.W.size), int((self.W != 0).sum())

    def to_dict(self):
        return {
            'n_in': self.n_in, 'n_out': self.n_out, 'total': self.total,
            'out_idx': list(self.out_idx),
            'W': self.W.tolist(), 'Win': self.Win.tolist(),
            'bias': self.bias.tolist(),
        }


# ============================================================
#                     Evolution
# ============================================================
def evolve(problem, max_gens, pop_size=300, verbose=True):
    X = problem['inputs']
    Y = problem['targets']
    steps = problem['steps']
    n_in = problem['n_in']
    n_out = problem['n_out']
    total0 = problem['init_total']
    POWERS = problem['POWERS']
    DENOM = problem['DENOM']
    kind = problem['kind']
    snapshot_gens = problem['snapshot_gens']

    pop = [ModNet(n_in, n_out, total0) for _ in range(pop_size)]
    fits_cache = [None] * pop_size
    history = {'gen': [], 'fit': [], 'rec': [], 'size': [], 'wires': []}
    snapshots = {}
    snapshot_saved = set()

    def eval_pop(indices):
        if not indices:
            return
        by_size = {}
        for k, i in enumerate(indices):
            by_size.setdefault(pop[i].total, []).append(k)
        B = X.shape[0]
        for size, local in by_size.items():
            G = len(local)
            N = size
            W_s = np.stack([pop[indices[k]].W for k in local])
            Win_s = np.stack([pop[indices[k]].Win for k in local])
            b_s = np.stack([pop[indices[k]].bias for k in local])
            oi_arr = np.array([pop[indices[k]].out_idx for k in local],
                              dtype=np.int32)
            const = np.matmul(Win_s, X.T) + b_s[:, :, None]
            const = np.transpose(const, (0, 2, 1))
            Wt = np.transpose(W_s, (0, 2, 1))
            state = np.zeros((G, B, N), dtype=np.float32)
            for _ in range(steps):
                z = np.matmul(state, Wt) + const
                state = (z >= 0).astype(np.float32)
            idx_b = np.broadcast_to(oi_arr[:, None, :], (G, B, oi_arr.shape[1]))
            out = np.take_along_axis(state, idx_b, axis=2)

            # Scalar for regression/single-output; bit-level for multi-bit boolean
            if Y.ndim == 1:
                preds = np.tensordot(out, POWERS, axes=([2], [0])) / DENOM
                diffs = (preds - Y[None, :]) ** 2
                fits = -np.mean(diffs, axis=1)
            else:
                out_trimmed = out[:, :, :n_out]
                diffs = (out_trimmed - Y[None, :, :]) ** 2
                fits = -np.mean(diffs, axis=(1, 2))

            for k, val in zip(local, fits):
                fits_cache[indices[k]] = float(val)

    best_fit = -1e9
    best_net = None
    t0 = time.time()
    plateau = 0
    sigma = 0.3
    elite_size = pop_size // 5

    for gen in range(max_gens):
        if gen in snapshot_gens and best_net is not None and gen not in snapshot_saved:
            snapshots[gen] = best_net.clone()
            snapshot_saved.add(gen)

        to_eval = [i for i in range(pop_size) if fits_cache[i] is None]
        eval_pop(to_eval)

        fits_arr = np.array(fits_cache, dtype=np.float64)
        order = np.argsort(fits_arr)[::-1]
        top_fit = float(fits_arr[order[0]])

        if top_fit > best_fit + 1e-5:
            best_fit = top_fit
            best_net = pop[int(order[0])].clone()
            plateau = 0
            sigma = 0.3
        else:
            plateau += 1
            if plateau > 25:
                sigma = min(1.5, sigma * 1.25)
                plateau = 0

        # Ensure generation 0 is always captured (top-of-loop check
        # cannot fire because best_net is None at that point).
        if (gen == 0 and 0 in snapshot_gens
                and 0 not in snapshot_saved
                and best_net is not None):
            snapshots[0] = best_net.clone()
            snapshot_saved.add(0)

        rec, ws, wnz = best_net.stats()
        history['gen'].append(gen)
        history['fit'].append(best_fit)
        history['rec'].append(rec)
        history['size'].append(best_net.total)
        history['wires'].append(wnz)

        # Birth
        if gen > 0 and gen % 100 == 0:
            born = 0
            for k in order[:3]:
                k = int(k)
                child = pop[k].birth()
                if child is not None:
                    pop[k] = child
                    fits_cache[k] = None
                    born += 1
            if verbose and born > 0:
                print(f"    gen {gen:4d}   🐣 born {born}   "
                      f"sizes={[pop[int(k)].total for k in order[:3]]}")

        # Prune
        if gen > 0 and gen % 20 == 0:
            total_removed = 0
            accepted = 0
            details = []
            for k in order[:elite_size]:
                k = int(k)
                res = pop[k].try_prune(X, Y, steps, POWERS, DENOM, kind,
                                       old_fit=float(fits_arr[k]))
                if res is not None:
                    cand, removed, nf = res
                    pop[k] = cand
                    fits_cache[k] = nf
                    fits_arr[k] = nf
                    total_removed += removed
                    accepted += 1
                    details.append(f"{pop[k].total}")
            if verbose and accepted > 0 and total_removed > 0:
                print(f"    gen {gen:4d}   ✂️ pruned {total_removed} nodes   "
                      f"from {accepted} elite   sizes now: {details[:5]}")

            new_top = int(np.argmax(fits_arr))
            if fits_arr[new_top] > best_fit + 1e-5:
                best_fit = float(fits_arr[new_top])
                best_net = pop[new_top].clone()

        if best_fit > -0.0005:
            if gen not in snapshot_saved:
                snapshots[gen] = best_net.clone()
            if verbose:
                print(f"    gen {gen:4d}   🎯 SOLVED   fit={best_fit:+.5f}   "
                      f"size={best_net.total}")
            break

        if verbose and gen % 200 == 0 and gen > 0:
            print(f"    gen {gen:4d}  fit={best_fit:+.5f}  "
                  f"size={best_net.total}  rec={rec}  wires={wnz}")

        order = np.argsort(fits_arr)[::-1]
        elite = [pop[int(i)] for i in order[:elite_size]]
        new_pop = list(elite)
        new_cache = [fits_cache[int(i)] for i in order[:elite_size]]
        while len(new_pop) < pop_size:
            rr = random.random()
            if rr < 0.30 and len(elite) >= 2:
                p1, p2 = random.sample(elite, 2)
                child = p1.crossover(p2)
                if child is None:
                    child = p1.mutate(random.randint(1, 3), sigma=sigma)
                else:
                    child = child.mutate(1, sigma=sigma * 0.5)
            elif rr < 0.95:
                child = random.choice(elite).mutate(
                    random.randint(1, 4), sigma=sigma)
            else:
                child = ModNet(n_in, n_out, total0)
            new_pop.append(child)
            new_cache.append(None)
        pop = new_pop
        fits_cache = new_cache

    if max_gens - 1 not in snapshot_saved:
        snapshots[max_gens - 1] = best_net.clone()

    elapsed = time.time() - t0
    return best_net, best_fit, history, snapshots, elapsed


# ============================================================
#                SPRING GRAPH
# ============================================================
def draw_net_spring(net, path, title='', fit=None):
    if not HAS_NX:
        return

    G = nx.DiGraph()
    out_set = set(net.out_idx)

    for k in range(net.n_in):
        G.add_node(f'in{k}', kind='input', idx=k)

    for i in range(net.total):
        kind = 'output' if i in out_set else 'hidden'
        G.add_node(i, kind=kind, idx=i)

    for i in range(net.total):
        for k in range(net.n_in):
            w = net.Win[i, k]
            if abs(w) < 0.15:
                continue
            G.add_edge(f'in{k}', i, weight=w, kind='input')

    for i in range(net.total):
        for j in range(net.total):
            w = net.W[i, j]
            if abs(w) < 0.15:
                continue
            G.add_edge(j, i, weight=w, kind='internal')

    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=80)

    fig, ax = plt.subplots(figsize=(13, 10), dpi=120)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    input_nodes = [n for n in G.nodes if G.nodes[n].get('kind') == 'input']
    hidden_nodes = [n for n in G.nodes if G.nodes[n].get('kind') == 'hidden']
    output_nodes = [n for n in G.nodes if G.nodes[n].get('kind') == 'output']

    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                           node_color=PALETTE_INPUT,
                           node_shape='s', node_size=350,
                           edgecolors='black', linewidths=1.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes,
                           node_color=PALETTE_HIDDEN,
                           node_shape='o', node_size=280,
                           edgecolors='black', linewidths=1.0, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes,
                           node_color=PALETTE_OUTPUT,
                           node_shape='o', node_size=340,
                           edgecolors='black', linewidths=2.0, ax=ax)

    input_edges = [(u, v) for u, v, d in G.edges(data=True)
                   if d['kind'] == 'input']
    internal_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d['kind'] == 'internal' and u != v]

    nx.draw_networkx_edges(G, pos, edgelist=input_edges,
                           edge_color='#0a8a3a', width=0.9,
                           alpha=0.5, arrows=True, arrowsize=8, ax=ax,
                           connectionstyle='arc3,rad=0.05')
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges,
                           edge_color='#1a5276', width=1.1,
                           alpha=0.55, arrows=True, arrowsize=9, ax=ax,
                           connectionstyle='arc3,rad=0.05')

    self_loops = [(i, i) for i in range(net.total)
                  if abs(net.W[i, i]) > 0.15]
    if self_loops:
        nx.draw_networkx_edges(G, pos, edgelist=self_loops,
                               edge_color='#cc0000', width=1.6,
                               alpha=0.85, arrows=True, arrowsize=11, ax=ax,
                               connectionstyle='arc3,rad=1.5')

    labels = {f'in{k}': f'x{k}' for k in range(net.n_in)}
    for i in range(net.total):
        labels[i] = f'n{i}'
    nx.draw_networkx_labels(G, pos, labels, font_size=9,
                            font_weight='bold', ax=ax)

    title_str = title
    if fit is not None:
        title_str += f'   (fit = {fit:+.5f})'
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.axis('off')

    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=PALETTE_INPUT, label=f'input ({net.n_in})'),
        mpatches.Patch(color=PALETTE_HIDDEN,
                       label=f'hidden ({net.total - len(out_set)})'),
        mpatches.Patch(color=PALETTE_OUTPUT,
                       label=f'output ({len(out_set)})'),
        mpatches.Patch(color='#cc0000', label='self-loop'),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path, format='png', dpi=120, bbox_inches='tight',
                facecolor='#fafafa')
    plt.close(fig)


# ============================================================
#                  Benchmark Problems
# ============================================================
def plot_curve(history, path, title=""):
    """Plot training curves (fitness, size, recurrence, wires)."""
    if not HAS_NX:
        return
    if not history.get("gen"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    gens = history["gen"]
    fits = history["fit"]
    sizes = history["size"]
    recs = history["rec"]
    wires = history["wires"]

    axes[0, 0].plot(gens, fits, color="#1a5276", lw=1.6)
    axes[0, 0].set_title("Fitness", fontweight="bold")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Best fitness")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(gens, sizes, color="#2E86AB", lw=1.6)
    axes[0, 1].set_title("Network size", fontweight="bold")
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Nodes")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(gens, recs, color="#C73E1D", lw=1.6)
    axes[1, 0].set_title("Recurrent self-loops", fontweight="bold")
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(gens, wires, color="#0a8a3a", lw=1.6)
    axes[1, 1].set_title("Active connections", fontweight="bold")
    axes[1, 1].set_xlabel("Generation")
    axes[1, 1].set_ylabel("Wires")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, format="png", dpi=120, bbox_inches="tight",
                facecolor="#fafafa")
    plt.close(fig)


def encode(x, bits):
    v = int(round(x * (2 ** bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]


def all_binary_inputs(n):
    return [tuple((i >> k) & 1 for k in range(n)) for i in range(2 ** n)]


def make_xor2():
    data = all_binary_inputs(2)
    X = np.array(data, dtype=np.float32)
    Y = np.array([a ^ b for a, b in data], dtype=np.float32)
    return dict(name='xor2', desc='2-bit XOR', n_in=2, n_out=1,
                steps=4, init_total=26, inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=400, snapshot_gens=[0, 100, 300, 399])


def make_xor3():
    data = all_binary_inputs(3)
    X = np.array(data, dtype=np.float32)
    Y = np.array([a ^ b ^ c for a, b, c in data], dtype=np.float32)
    return dict(name='xor3', desc='3-bit XOR', n_in=3, n_out=1,
                steps=5, init_total=26, inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=600, snapshot_gens=[0, 150, 400, 599])


def make_parity4():
    data = all_binary_inputs(4)
    X = np.array(data, dtype=np.float32)
    Y = np.array([sum(d) % 2 for d in data], dtype=np.float32)
    return dict(name='parity4', desc='Parity-4', n_in=4, n_out=1,
                steps=6, init_total=26, inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=800, snapshot_gens=[0, 200, 500, 799])


def make_parity6():
    data = all_binary_inputs(6)
    X = np.array(data, dtype=np.float32)
    Y = np.array([sum(d) % 2 for d in data], dtype=np.float32)
    return dict(name='parity6', desc='Parity-6', n_in=6, n_out=1,
                steps=8, init_total=26, inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=1500, snapshot_gens=[0, 400, 900, 1499])


def make_majority3():
    data = all_binary_inputs(3)
    X = np.array(data, dtype=np.float32)
    Y = np.array([1 if sum(d) >= 2 else 0 for d in data], dtype=np.float32)
    return dict(name='majority3', desc='Majority of 3',
                n_in=3, n_out=1, steps=5, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=500, snapshot_gens=[0, 150, 350, 499])


def make_majority5():
    data = all_binary_inputs(5)
    X = np.array(data, dtype=np.float32)
    Y = np.array([1 if sum(d) >= 3 else 0 for d in data], dtype=np.float32)
    return dict(name='majority5', desc='Majority of 5',
                n_in=5, n_out=1, steps=7, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=900, snapshot_gens=[0, 250, 600, 899])


def make_full_adder():
    data = all_binary_inputs(3)
    X = np.array(data, dtype=np.float32)
    Y_list = []
    for a, b, cin in data:
        s = (a + b + cin) & 1
        c = (a + b + cin) >> 1
        Y_list.append([s, c])
    Y = np.array(Y_list, dtype=np.float32)
    return dict(name='full_adder', desc='Full Adder (sum, carry)',
                n_in=3, n_out=2, steps=6, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0, 2.0], dtype=np.float32), DENOM=3.0,
                max_gens=800, snapshot_gens=[0, 200, 500, 799])


def make_adder2bit():
    data = all_binary_inputs(4)
    X = np.array(data, dtype=np.float32)
    Y_list = []
    for a0, a1, b0, b1 in data:
        total = a0 + 2*a1 + b0 + 2*b1
        s0 = total & 1
        s1 = (total >> 1) & 1
        cout = (total >> 2) & 1
        Y_list.append([s0, s1, cout])
    Y = np.array(Y_list, dtype=np.float32)
    return dict(name='adder2bit', desc='2-bit Adder (s0, s1, cout)',
                n_in=4, n_out=3, steps=8, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0, 2.0, 4.0], dtype=np.float32),
                DENOM=7.0,
                max_gens=1200, snapshot_gens=[0, 300, 700, 1199])


def make_comparator():
    data = all_binary_inputs(2)
    X = np.array(data, dtype=np.float32)
    Y_list = []
    for a, b in data:
        gt = 1 if a > b else 0
        eq = 1 if a == b else 0
        lt = 1 if a < b else 0
        Y_list.append([gt, eq, lt])
    Y = np.array(Y_list, dtype=np.float32)
    return dict(name='comparator', desc='Comparator (gt, eq, lt)',
                n_in=2, n_out=3, steps=6, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0, 2.0, 4.0], dtype=np.float32),
                DENOM=7.0,
                max_gens=600, snapshot_gens=[0, 150, 400, 599])


def make_mux4to1():
    data = all_binary_inputs(6)
    X = np.array(data, dtype=np.float32)
    Y_list = []
    for d0, d1, d2, d3, s0, s1 in data:
        idx = s0 + 2*s1
        out = [d0, d1, d2, d3][idx]
        Y_list.append(out)
    Y = np.array(Y_list, dtype=np.float32)
    return dict(name='mux4to1', desc='4-to-1 MUX',
                n_in=6, n_out=1, steps=8, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=1200, snapshot_gens=[0, 300, 700, 1199])


def make_sort4():
    data = all_binary_inputs(4)
    X = np.array(data, dtype=np.float32)
    Y_list = []
    for bits in data:
        s = sorted(bits)
        Y_list.append(s)
    Y = np.array(Y_list, dtype=np.float32)
    return dict(name='sort4', desc='4-bit Sorting Network',
                n_in=4, n_out=4, steps=8, init_total=26,
                inputs=X, targets=Y, kind='boolean',
                POWERS=np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32),
                DENOM=15.0,
                max_gens=1200, snapshot_gens=[0, 300, 700, 1199])


def make_sin2pi():
    N = 32
    xs = np.linspace(0, 1, N)
    X = np.array([encode(x, 4) for x in xs], dtype=np.float32)
    Y = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)
    return dict(name='sin2pi', desc='sin(2πx)',
                n_in=4, n_out=8, steps=5, init_total=26,
                inputs=X, targets=Y, kind='regression',
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 400, 900, 1499])


def make_sin4pi():
    N = 32
    xs = np.linspace(0, 1, N)
    X = np.array([encode(x, 4) for x in xs], dtype=np.float32)
    Y = (np.sin(xs * 4 * np.pi) * 0.5 + 0.5).astype(np.float32)
    return dict(name='sin4pi', desc='sin(4πx)',
                n_in=4, n_out=8, steps=6, init_total=26,
                inputs=X, targets=Y, kind='regression',
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 400, 900, 1499])


def make_circle():
    n_side = 5
    xs = np.linspace(0, 1, n_side)
    grid = [(x, y) for x in xs for y in xs]
    X = np.array([encode(x, 4) + encode(y, 4) for x, y in grid],
                 dtype=np.float32)
    Y = np.array([0.5 + 0.5 * (x * x + y * y) / 2 for x, y in grid],
                 dtype=np.float32)
    return dict(name='circle', desc='x² + y²',
                n_in=8, n_out=8, steps=6, init_total=26,
                inputs=X, targets=Y, kind='regression',
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1200,
                snapshot_gens=[0, 300, 700, 1199])


def make_gaussian():
    N = 25
    xs = np.linspace(-1, 1, N)
    X = np.array([encode((x + 1) / 2, 4) for x in xs], dtype=np.float32)
    Y = np.exp(-xs ** 2).astype(np.float32)
    return dict(name='gaussian', desc='Gaussian exp(-x²)',
                n_in=4, n_out=8, steps=5, init_total=26,
                inputs=X, targets=Y, kind='regression',
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 400, 900, 1499])


def make_multiplication():
    n_side = 6
    xs = np.linspace(0, 1, n_side)
    grid = [(x, y) for x in xs for y in xs]
    X = np.array([encode(x, 4) + encode(y, 4) for x, y in grid],
                 dtype=np.float32)
    Y = np.array([x * y for x, y in grid], dtype=np.float32)
    return dict(name='multiply', desc='x · y',
                n_in=8, n_out=8, steps=6, init_total=26,
                inputs=X, targets=Y, kind='regression',
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 400, 900, 1499])


# ============================================================
#                Runner
# ============================================================
def run_problem(problem, idx_global):
    name = problem['name']
    print(f"\n{'=' * 60}")
    print(f"  [{idx_global}] {name}: {problem['desc']}")
    print(f"  n_in={problem['n_in']}  n_out={problem['n_out']}  "
          f"steps={problem['steps']}  init_total={problem['init_total']}")
    print(f"{'=' * 60}")

    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)

    net, fit, hist, snaps, ev_time = evolve(problem, problem['max_gens'])
    rec, ws, wnz = net.stats()

    preds = net.predict_values(problem['inputs'], problem['steps'],
                               problem['POWERS'], problem['DENOM'])
    Y = problem['targets']

    if problem['kind'] == 'boolean':
        if Y.ndim == 1:
            pred_bits = (preds >= 0.5).astype(int)
            correct = int((pred_bits == Y.astype(int)).sum())
            accuracy = correct / len(Y)
            metric_str = f'acc={correct}/{len(Y)}'
        else:
            out_bits = net.predict_bits(problem['inputs'], problem['steps'])
            pred_bits = (out_bits >= 0.5).astype(int)
            sample_correct = (pred_bits == Y.astype(int)).all(axis=1)
            correct = int(sample_correct.sum())
            accuracy = correct / len(Y)
            metric_str = f'acc={correct}/{len(Y)} samples'
        rmse = math.sqrt(max(0, -fit))
    else:
        rmse = math.sqrt(max(0, -fit))
        accuracy = None
        metric_str = f'RMSE={rmse:.4f}'

    with open(out_dir / 'best.json', 'w') as f:
        json.dump(net.to_dict(), f, indent=2)

    with open(out_dir / 'curve.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gen', 'fit', 'rec', 'size', 'wires'])
        for g, ft, r, s, wr in zip(hist['gen'], hist['fit'],
                                    hist['rec'], hist['size'],
                                    hist['wires']):
            w.writerow([g, f'{ft:.6f}', r, s, wr])

    plot_curve(hist, out_dir / 'curve.png',
               title=f'{name} - {problem["desc"]}')

    draw_net_spring(net, out_dir / 'best_spring.png',
                    title=f'{name} - {problem["desc"]}', fit=fit)

    for gen, snap_net in sorted(snaps.items()):
        draw_net_spring(snap_net,
                        out_dir / f'snapshot_gen_{gen:04d}.png',
                        title=f'{name} @ gen {gen}')

    print(f"  --> fit={fit:+.5f}  {metric_str}  "
          f"size={net.total}  rec={rec}  wires={wnz}")
    print(f"  time: {ev_time:.1f}s")

    solved = False
    if accuracy is not None:
        solved = (accuracy == 1.0)
    else:
        solved = (rmse < 0.05)

    return {
        'name': name, 'desc': problem['desc'],
        'n_in': problem['n_in'], 'n_out': problem['n_out'],
        'kind': problem['kind'],
        'best_fit': fit, 'rmse': rmse, 'accuracy': accuracy,
        'size': net.total, 'rec': rec, 'wires': wnz,
        'time': ev_time,
        'gens_run': hist['gen'][-1] if hist['gen'] else 0,
        'status': 'SOLVED' if solved else 'PARTIAL',
    }


def main():
    print('=' * 60)
    print('  ModNet Benchmark Suite v2 + Spring Graph')
    print(f'  seed = {SEED}  (per-problem seed reset)')
    print('=' * 60)

    problems = [
        make_xor2(),
        make_xor3(),
        make_parity4(),
        make_parity6(),
        make_majority3(),
        make_majority5(),
        make_full_adder(),
        make_adder2bit(),
        make_comparator(),
        make_mux4to1(),
        make_sort4(),
        make_sin2pi(),
        make_sin4pi(),
        make_circle(),
        make_gaussian(),
        make_multiplication(),
    ]

    results = []
    t0 = time.time()
    for i, prob in enumerate(problems, 1):
        # ★ Seed reset per problem for fair comparison ★
        random.seed(SEED)
        np.random.seed(SEED)

        try:
            r = run_problem(prob, i)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {prob['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total = time.time() - t0

    # Summary
    with open(OUT / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'desc', 'kind', 'n_in', 'n_out',
                    'best_fit', 'rmse', 'accuracy', 'size', 'rec',
                    'wires', 'time', 'gens_run', 'status'])
        for r in results:
            w.writerow([r['name'], r['desc'], r['kind'],
                        r['n_in'], r['n_out'],
                        f"{r['best_fit']:.6f}", f"{r['rmse']:.6f}",
                        (f"{r['accuracy']:.4f}"
                         if r['accuracy'] is not None else ''),
                        r['size'], r['rec'], r['wires'],
                        f"{r['time']:.1f}", r['gens_run'], r['status']])

    lines = ['=' * 78,
             'BENCHMARK SUITE v2 SUMMARY',
             f'seed = {SEED}   total time = {total:.1f}s',
             '=' * 78, '']
    lines.append(f'{"problem":<12s} {"kind":<11s} {"fit":>10s} '
                 f'{"acc/RMSE":>11s} {"size":>5s} {"rec":>4s} '
                 f'{"wires":>6s} {"time":>7s} {"status":<8s}')
    lines.append('-' * 78)
    for r in results:
        if r['accuracy'] is not None:
            metric = f"{r['accuracy']*100:.0f}%"
        else:
            metric = f"{r['rmse']:.4f}"
        lines.append(f"{r['name']:<12s} {r['kind']:<11s} "
                     f"{r['best_fit']:>+10.5f} {metric:>11s} "
                     f"{r['size']:>5d} {r['rec']:>4d} "
                     f"{r['wires']:>6d} {r['time']:>6.1f}s "
                     f"{r['status']:<8s}")

    with open(OUT / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    solved = [r for r in results if r['status'] == 'SOLVED']
    partial = [r for r in results if r['status'] == 'PARTIAL']

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")
    print(f"  Total:   {total:.1f}s")
    print(f"  Solved:  {len(solved)}/{len(results)}")
    print(f"  Partial: {len(partial)}/{len(results)}")
    print(f"  Output:  {OUT}/")
    print(f"  Summary: {OUT}/summary.txt")


if __name__ == '__main__':
    main()
