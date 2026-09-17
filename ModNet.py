"""
Comprehensive experiment framework for ModNet
==============================================
- Boolean problems: Y is (N,1) target bit, fitness on values.
- Regression problems: Y is (N,) target values in [0,1], fitness on values.

Outputs per problem:
  results/<name>/
    best.json          - full network state
    best.svg           - graphical rendering
    best.txt           - plain-text network description
    curve.csv          - fitness history
    curve.svg          - learning curve plot
    curve.txt          - ASCII curve
    samples.csv        - predictions on all training samples
    snapshots/
      gen_XXXX.svg     - network at selected generations
    snapshots.txt      - text descriptions of snapshots
"""
import os, json, csv, time, random, math
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT = Path('results')
OUT.mkdir(exist_ok=True)

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
           '#6A994E', '#BC4B51', '#5B8E7D', '#F4A259']


# ============================================================
#                     ModNet
# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module, out_size):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.out_size = out_size
        self._alloc()
        self._init()

    def _alloc(self):
        self.total = sum(self.mods)
        self.W = np.zeros((self.total, self.total), dtype=np.float32)
        self.Win = np.zeros((self.total, self.n_in), dtype=np.float32)
        self.bias = np.zeros(self.total, dtype=np.float32)
        self._rebuild_mod_map()

    def _rebuild_mod_map(self):
        self._mod_map = np.zeros(self.total, dtype=np.int32)
        s = 0
        for m, sz in enumerate(self.mods):
            for i in range(s, s + sz):
                self._mod_map[i] = m
            s += sz
        out_all = [i for i in range(self.total) if self._mod_map[i] == self.out_mod]
        self.out_idx = np.array(out_all[:self.out_size], dtype=np.int32)
        if len(self.out_idx) < self.out_size:
            extra = [i for i in range(self.total)
                     if self._mod_map[i] != self.out_mod][:self.out_size - len(self.out_idx)]
            self.out_idx = np.array(list(self.out_idx) + extra, dtype=np.int32)

    def _mod_of(self, i):
        return int(self._mod_map[i])

    def _init(self):
        for i in range(self.total):
            for _ in range(random.randint(1, 3)):
                if self._mod_of(i) == self.out_mod or random.random() > 0.3:
                    self.W[i, random.randrange(self.total)] += random.gauss(0, 1)
                else:
                    self.Win[i, random.randrange(self.n_in)] += random.gauss(0, 1)
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
        out = self.forward_batch(X, steps)
        return (out @ POWERS) / DENOM

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
        preds = self.predict_values(X, steps, POWERS, DENOM)
        return float(-np.mean((preds - Y) ** 2))

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = list(self.mods)
        n.out_mod = self.out_mod
        n.out_size = self.out_size
        n.total = self.total
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        n._mod_map = self._mod_map.copy()
        n.out_idx = self.out_idx.copy()
        return n

    def mutate(self, rate=2, sigma=0.5):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            r = random.random()
            if r < 0.40:
                new.W[i, random.randrange(new.total)] += random.gauss(0, sigma)
            elif r < 0.60:
                new.bias[i] += random.gauss(0, sigma)
            elif r < 0.80:
                new.Win[i, random.randrange(new.n_in)] += random.gauss(0, sigma)
            elif r < 0.92:
                nz = np.nonzero(new.W[i])[0]
                if len(nz) > 1:
                    new.W[i, int(random.choice(nz))] = 0
            else:
                new.W[i, random.randrange(new.total)] = random.gauss(0, 1)
        return new

    def birth(self, max_total=60, max_mod=15):
        if self.total >= max_total:
            return None
        candidates = [m for m in range(len(self.mods)) if self.mods[m] < max_mod]
        if not candidates:
            return None
        weights = [1.0 / (self.mods[m] + 1) for m in candidates]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        r = random.random()
        acc = 0
        chosen_mod = candidates[-1]
        for m, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                chosen_mod = m
                break
        insert_pos = sum(self.mods[:chosen_mod + 1])
        N = self.total
        W_new = np.zeros((N + 1, N + 1), dtype=np.float32)
        Win_new = np.zeros((N + 1, self.n_in), dtype=np.float32)
        bias_new = np.zeros(N + 1, dtype=np.float32)
        for i in range(N):
            ni = i if i < insert_pos else i + 1
            bias_new[ni] = self.bias[i]
            Win_new[ni] = self.Win[i]
            for j in range(N):
                nj = j if j < insert_pos else j + 1
                W_new[ni, nj] = self.W[i, j]
        for _ in range(random.randint(1, 3)):
            src = random.randrange(N + 1)
            if src != insert_pos:
                W_new[insert_pos, src] = random.gauss(0, 1)
        for _ in range(random.randint(0, 2)):
            Win_new[insert_pos, random.randrange(self.n_in)] = random.gauss(0, 1)
        bias_new[insert_pos] = random.gauss(0, 1)
        new = self.clone()
        new.mods = list(self.mods)
        new.mods[chosen_mod] += 1
        new.total = N + 1
        new.W = W_new
        new.Win = Win_new
        new.bias = bias_new
        new._rebuild_mod_map()
        return new

    def crossover(self, other):
        if self.total != other.total or self.mods != other.mods:
            return None
        new = self.clone()
        for i in range(new.total):
            if random.random() < 0.5:
                new.W[i] = other.W[i].copy()
                new.Win[i] = other.Win[i].copy()
                new.bias[i] = other.bias[i]
        return new

    def _prune_inplace(self, X, steps, w_thresh=0.04, act_lo=0.02,
                       act_hi=0.98, min_size=2):
        act = self.activity(X, steps)
        self.W[np.abs(self.W) < w_thresh] = 0
        self.Win[np.abs(self.Win) < w_thresh] = 0
        alive = np.ones(self.total, dtype=bool)
        alive &= ~((act < act_lo) | (act > act_hi))
        in_from_W = np.abs(self.W).sum(axis=0)
        in_from_in = np.abs(self.Win).sum(axis=1)
        out_to = np.abs(self.W).sum(axis=1)
        alive &= ((in_from_W + in_from_in) > 0) | (out_to > 0)
        per_mod_keep = []
        s = 0
        for m, sz in enumerate(self.mods):
            idx = list(range(s, s + sz))
            keep = [i for i in idx if alive[i]]
            if len(keep) < min_size:
                sorted_idx = sorted(idx, key=lambda i: -abs(act[i] - 0.5))
                keep = sorted_idx[:min_size]
            per_mod_keep.append(sorted(keep))
            s += sz
        if len(per_mod_keep[self.out_mod]) < self.out_size:
            s = sum(self.mods[:self.out_mod])
            per_mod_keep[self.out_mod] = list(range(s, s + self.mods[self.out_mod]))
        keep = []
        for m_idx in per_mod_keep:
            keep.extend(m_idx)
        keep = np.array(sorted(set(keep)), dtype=np.int32)
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
        self.mods = [len(m) for m in per_mod_keep]
        self.W = new_W
        self.Win = new_Win
        self.bias = new_bias
        self.total = new_total
        self._rebuild_mod_map()
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
        cross = sum(1 for i, j in zip(Wnz[0], Wnz[1])
                    if self._mod_map[i] != self._mod_map[j])
        rec = sum(1 for i, j in zip(Wnz[0], Wnz[1]) if i == j)
        return cross, rec, int(self.W.size), int((self.W != 0).sum())

    def to_dict(self):
        return {
            'n_in': self.n_in, 'mods': self.mods,
            'out_mod': self.out_mod, 'out_size': self.out_size,
            'total': self.total,
            'W': self.W.tolist(), 'Win': self.Win.tolist(),
            'bias': self.bias.tolist(),
            'out_idx': self.out_idx.tolist(),
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
    POWERS = problem['POWERS']
    DENOM = problem['DENOM']
    MODULE_SIZES = problem['MODULE_SIZES']
    OUT_MODULE = problem['OUT_MODULE']
    kind = problem['kind']
    snapshot_gens = problem['snapshot_gens']

    pop = [ModNet(n_in, MODULE_SIZES, OUT_MODULE, n_out) for _ in range(pop_size)]
    fits_cache = [None] * pop_size

    history = {'gen': [], 'fit': [], 'cross': [], 'rec': [],
               'size': [], 'wires': []}
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
            oi_s = np.stack([pop[indices[k]].out_idx for k in local])
            const = np.matmul(Win_s, X.T) + b_s[:, :, None]
            const = np.transpose(const, (0, 2, 1))
            Wt = np.transpose(W_s, (0, 2, 1))
            state = np.zeros((G, B, N), dtype=np.float32)
            for _ in range(steps):
                z = np.matmul(state, Wt) + const
                state = (z >= 0).astype(np.float32)
            idx_b = np.broadcast_to(oi_s[:, None, :], (G, B, oi_s.shape[1]))
            out = np.take_along_axis(state, idx_b, axis=2)
            preds = np.tensordot(out, POWERS, axes=([2], [0])) / DENOM
            diffs = (preds - Y[None, :]) ** 2
            fits = -np.mean(diffs, axis=1)
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

        c, r, ws, wnz = best_net.stats()
        history['gen'].append(gen)
        history['fit'].append(best_fit)
        history['cross'].append(c)
        history['rec'].append(r)
        history['size'].append(best_net.total)
        history['wires'].append(wnz)

        if gen > 0 and gen % 100 == 0:
            for k in order[:3]:
                k = int(k)
                child = pop[k].birth()
                if child is not None:
                    pop[k] = child
                    fits_cache[k] = None

        if gen > 0 and gen % 20 == 0:
            for k in order[:elite_size]:
                k = int(k)
                res = pop[k].try_prune(X, Y, steps, POWERS, DENOM, kind,
                                       old_fit=float(fits_arr[k]))
                if res is not None:
                    cand, _, nf = res
                    pop[k] = cand
                    fits_cache[k] = nf
                    fits_arr[k] = nf
            new_top = int(np.argmax(fits_arr))
            if fits_arr[new_top] > best_fit + 1e-5:
                best_fit = float(fits_arr[new_top])
                best_net = pop[new_top].clone()

        if best_fit > -0.0005:
            if gen not in snapshot_saved:
                snapshots[gen] = best_net.clone()
            break

        if verbose and gen % 200 == 0 and gen > 0:
            print(f"    gen {gen:4d}  fit={best_fit:+.5f}  "
                  f"size={best_net.total}  cross={c}  rec={r}")

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
                child = ModNet(n_in, MODULE_SIZES, OUT_MODULE, n_out)
            new_pop.append(child)
            new_cache.append(None)
        pop = new_pop
        fits_cache = new_cache

    if max_gens - 1 not in snapshot_saved:
        snapshots[max_gens - 1] = best_net.clone()

    elapsed = time.time() - t0
    return best_net, best_fit, history, snapshots, elapsed


# ============================================================
#                    Problems
# ============================================================
def encode(x, bits):
    v = int(round(x * (2 ** bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]


def make_xor2():
    data = [(a, b) for a in (0, 1) for b in (0, 1)]
    X = np.array(data, dtype=np.float32)
    Y = np.array([a ^ b for a, b in data], dtype=np.float32)
    return dict(name='xor2', desc='2-bit XOR',
                n_in=2, n_out=1, steps=4,
                inputs=X, targets=Y, kind='boolean',
                MODULE_SIZES=[6, 6, 6, 1], OUT_MODULE=3,
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=400, snapshot_gens=[0, 50, 150, 300, 399])


def make_parity4():
    X = np.array([[(i >> k) & 1 for k in range(4)] for i in range(16)],
                 dtype=np.float32)
    Y = np.array([bin(i).count('1') % 2 for i in range(16)], dtype=np.float32)
    return dict(name='parity4', desc='Parity of 4 bits',
                n_in=4, n_out=1, steps=6,
                inputs=X, targets=Y, kind='boolean',
                MODULE_SIZES=[6, 6, 6, 1], OUT_MODULE=3,
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=800, snapshot_gens=[0, 100, 300, 500, 799])


def make_parity6():
    X = np.array([[(i >> k) & 1 for k in range(6)] for i in range(64)],
                 dtype=np.float32)
    Y = np.array([bin(i).count('1') % 2 for i in range(64)], dtype=np.float32)
    return dict(name='parity6', desc='Parity of 6 bits',
                n_in=6, n_out=1, steps=8,
                inputs=X, targets=Y, kind='boolean',
                MODULE_SIZES=[6, 6, 6, 1], OUT_MODULE=3,
                POWERS=np.array([1.0], dtype=np.float32), DENOM=1.0,
                max_gens=1500, snapshot_gens=[0, 300, 700, 1100, 1499])


def make_sin2pi():
    N = 32
    xs = np.linspace(0, 1, N)
    X = np.array([encode(x, 4) for x in xs], dtype=np.float32)
    Y = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)
    return dict(name='sin2pi', desc='sin(2*pi*x)',
                n_in=4, n_out=8, steps=5,
                inputs=X, targets=Y, kind='regression',
                MODULE_SIZES=[6, 6, 6, 8], OUT_MODULE=3,
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 300, 700, 1100, 1499])


def make_sin4pi():
    N = 32
    xs = np.linspace(0, 1, N)
    X = np.array([encode(x, 4) for x in xs], dtype=np.float32)
    Y = (np.sin(xs * 4 * np.pi) * 0.5 + 0.5).astype(np.float32)
    return dict(name='sin4pi', desc='sin(4*pi*x)',
                n_in=4, n_out=8, steps=6,
                inputs=X, targets=Y, kind='regression',
                MODULE_SIZES=[6, 6, 6, 8], OUT_MODULE=3,
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1500,
                snapshot_gens=[0, 300, 700, 1100, 1499])


def make_circle():
    n_side = 5
    xs = np.linspace(0, 1, n_side)
    grid = [(x, y) for x in xs for y in xs]
    X = np.array([encode(x, 4) + encode(y, 4) for x, y in grid],
                 dtype=np.float32)
    Y = np.array([0.5 + 0.5 * (x * x + y * y) / 2 for x, y in grid],
                 dtype=np.float32)
    return dict(name='circle', desc='x^2 + y^2',
                n_in=8, n_out=8, steps=6,
                inputs=X, targets=Y, kind='regression',
                MODULE_SIZES=[6, 6, 6, 8], OUT_MODULE=3,
                POWERS=np.array([2 ** k for k in range(8)], dtype=np.float32),
                DENOM=255.0, max_gens=1200,
                snapshot_gens=[0, 250, 550, 850, 1199])


# ============================================================
#                SVG Rendering — Network
# ============================================================
def draw_net_svg(net, path, title='', fit=None):
    """
    Render a ModNet as a clean SVG with proper edge routing.
      - Edges clipped at node boundaries (no more 'coming from behind').
      - Input ports drawn as labeled squares on the left.
      - Arrowheads on every edge.
      - Recurrent self-loops drawn as arcs above nodes.
      - Positive/negative weights colored differently.
      - Legend at the bottom.
    """
    n_mods = len(net.mods)
    CS = 180
    PAD = 60
    TOP = 80
    BOTTOM = 80
    R = 16
    INPUT_COL_W = 80
    LEFT_MARGIN = PAD + INPUT_COL_W

    max_size = max(net.mods) if net.mods else 1
    row_h = 34
    H = max(460, TOP + max_size * row_h + BOTTOM)
    W = LEFT_MARGIN + n_mods * CS + PAD

    # ---------- Node positions ----------
    pos = {}
    for i in range(net.total):
        m = net._mod_of(i)
        m_start = sum(net.mods[:m])
        m_size = net.mods[m]
        idx_in_m = i - m_start
        x = LEFT_MARGIN + m * CS + CS // 2
        if m_size == 1:
            y = TOP + (H - TOP - BOTTOM) // 2
        else:
            span = H - TOP - BOTTOM - 2 * PAD
            y = TOP + PAD + idx_in_m * span / (m_size - 1)
        pos[i] = (x, y)

    # ---------- Input port positions ----------
    n_inputs = net.n_in
    avail_h = H - TOP - BOTTOM - 2 * PAD
    if n_inputs > 0:
        spacing = max(28.0, min(64.0, avail_h / max(n_inputs, 1)))
        total_h = (n_inputs - 1) * spacing
        y0 = TOP + PAD + (avail_h - total_h) / 2.0
        input_ports = {k: (PAD, y0 + k * spacing) for k in range(n_inputs)}
    else:
        input_ports = {}

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" style="background:#fafafa;font-family:monospace">')

    # ---------- Arrowhead markers ----------
    def marker_def(mid, color, size=6):
        return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="{size}" markerHeight="{size}" '
                f'orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>')

    s.append('<defs>')
    s.append(marker_def('arr-gray',   '#888888', 5))
    s.append(marker_def('arr-blue',   '#1a5276', 6))
    s.append(marker_def('arr-red',    '#cc0000', 6))
    s.append(marker_def('arr-green',  '#0a8a3a', 5))
    s.append(marker_def('arr-orange', '#b85a00', 5))
    s.append('</defs>')

    # ---------- Title ----------
    s.append(f'<text x="{PAD}" y="28" font-size="16" fill="black" '
             f'font-weight="bold">{title}</text>')
    if fit is not None:
        s.append(f'<text x="{PAD}" y="48" font-size="12" fill="#666">'
                 f'fitness = {fit:+.5f}</text>')

    # ---------- Module bands ----------
    for m in range(n_mods):
        x0 = LEFT_MARGIN + m * CS + 10
        bw = CS - 20
        y0 = TOP + PAD - 35
        bh = H - TOP - BOTTOM - 2 * PAD + 70
        col = PALETTE[m % len(PALETTE)]
        s.append(f'<rect x="{x0}" y="{y0}" width="{bw}" height="{bh}" '
                 f'fill="{col}" opacity="0.06" rx="8"/>')
        lbl = f'M{m}' + (' [OUT]' if m == net.out_mod else '')
        s.append(f'<text x="{x0+8}" y="{y0+15}" font-size="11" '
                 f'fill="{col}" font-weight="bold">{lbl}</text>')

    # ---------- Input port squares ----------
    for k in range(n_inputs):
        px, py = input_ports[k]
        s.append(f'<rect x="{px}" y="{py-11}" width="26" height="22" '
                 f'fill="#d6ecff" stroke="#0a8a3a" stroke-width="1.5" rx="3"/>')
        s.append(f'<text x="{px+13}" y="{py+5}" text-anchor="middle" '
                 f'font-size="11" fill="#0a8a3a" font-weight="bold">'
                 f'x{k}</text>')

    # ---------- Input wires (from port to node boundary) ----------
    for i in range(net.total):
        for k in range(net.n_in):
            w = net.Win[i, k]
            if abs(w) < 0.15:
                continue
            sx, sy = input_ports[k]
            tx, ty = pos[i]
            start_x = sx + 26
            start_y = sy
            dx = tx - start_x
            dy = ty - start_y
            d = (dx * dx + dy * dy) ** 0.5
            if d < R + 6:
                continue
            ux, uy = dx / d, dy / d
            end_x = tx - ux * (R + 5)
            end_y = ty - uy * (R + 5)
            op = min(1.0, abs(w) / 1.5)
            col = '#0a8a3a' if w > 0 else '#b85a00'
            mid = 'arr-green' if w > 0 else 'arr-orange'
            s.append(f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" '
                     f'x2="{end_x:.1f}" y2="{end_y:.1f}" '
                     f'stroke="{col}" stroke-width="1.1" '
                     f'opacity="{op:.2f}" marker-end="url(#{mid})"/>')

    # ---------- Recurrent self-loops ----------
    for i in range(net.total):
        w = net.W[i, i]
        if abs(w) < 0.15:
            continue
        x, y = pos[i]
        arc_h = 34
        # Start/end at the top of the node circle
        dx_off = 5
        dy_off = (R * R - dx_off * dx_off) ** 0.5
        sx = x - dx_off
        sy = y - dy_off
        ex = x + dx_off
        ey = y - dy_off
        d = (f'M {sx:.1f} {sy:.1f} '
             f'C {x - 26:.1f} {y - R - arc_h:.1f}, '
             f'{x + 26:.1f} {y - R - arc_h:.1f}, '
             f'{ex:.1f} {ey:.1f}')
        s.append(f'<path d="{d}" fill="none" stroke="#cc0000" '
                 f'stroke-width="1.6" marker-end="url(#arr-red)"/>')
        s.append(f'<text x="{x+32}" y="{y - R - arc_h + 8}" '
                 f'font-size="9" fill="#cc0000">{w:+.2f}</text>')

    # ---------- Internal edges ----------
    for i in range(net.total):
        for j in range(net.total):
            if i == j:
                continue
            w = net.W[i, j]
            if abs(w) < 0.15:
                continue
            x1, y1 = pos[j]
            x2, y2 = pos[i]
            dx = x2 - x1
            dy = y2 - y1
            d = (dx * dx + dy * dy) ** 0.5
            if d < 2 * R + 8:
                continue
            ux, uy = dx / d, dy / d
            start_x = x1 + ux * R
            start_y = y1 + uy * R
            end_x = x2 - ux * (R + 5)
            end_y = y2 - uy * (R + 5)
            m_s = net._mod_of(j)
            m_d = net._mod_of(i)
            if m_s == m_d:
                col = '#888888'
                wid = 0.8
                mid = 'arr-gray'
            else:
                col = '#1a5276'
                wid = 1.3
                mid = 'arr-blue'
            op = min(1.0, abs(w) / 1.5)
            s.append(f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" '
                     f'x2="{end_x:.1f}" y2="{end_y:.1f}" '
                     f'stroke="{col}" stroke-width="{wid}" '
                     f'opacity="{op:.2f}" marker-end="url(#{mid})"/>')

    # ---------- Nodes (drawn on top) ----------
    for i in range(net.total):
        x, y = pos[i]
        m = net._mod_of(i)
        col = PALETTE[m % len(PALETTE)]
        is_out = i in net.out_idx
        stroke = '#cc0000' if is_out else 'black'
        sw = 3 if is_out else 1.5
        s.append(f'<circle cx="{x}" cy="{y}" r="{R}" fill="{col}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')
        s.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" '
                 f'font-size="10" fill="white" font-weight="bold">'
                 f'{i}</text>')

    # ---------- Output markers ----------
    for i in net.out_idx:
        x, y = pos[i]
        s.append(f'<line x1="{x+R+4}" y1="{y}" x2="{x+R+26}" y2="{y}" '
                 f'stroke="#cc0000" stroke-width="2.5"/>')
        s.append(f'<text x="{x+R+30}" y="{y+4}" font-size="10" '
                 f'fill="#cc0000" font-weight="bold">out</text>')

    # ---------- Legend ----------
    legend_y = H - 22
    items = [
        ('#888888', 'intra-module'),
        ('#1a5276', 'cross-module'),
        ('#cc0000', 'self-loop'),
        ('#0a8a3a', 'input (+)'),
        ('#b85a00', 'input (\u2212)'),
    ]
    x_leg = PAD
    for col, label in items:
        s.append(f'<line x1="{x_leg}" y1="{legend_y}" '
                 f'x2="{x_leg+22}" y2="{legend_y}" '
                 f'stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<text x="{x_leg+28}" y="{legend_y+4}" font-size="10" '
                 f'fill="#333">{label}</text>')
        x_leg += 140

    s.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(s))


# ============================================================
#                SVG Rendering — Learning Curve
# ============================================================
def draw_curve_svg(history, path, title=''):
    W = 720
    H = 420
    PAD_L = 70
    PAD_R = 40
    PAD_T = 50
    PAD_B = 60

    gens = history['gen']
    fits = history['fit']
    if not gens:
        return
    gmin = min(gens)
    gmax = max(gens) if max(gens) > 0 else 1
    fmin = min(fits)
    fmax = max(fits)
    if fmax - fmin < 1e-6:
        fmin -= 0.001
        fmax += 0.001

    def px(g):
        return PAD_L + (g - gmin) / max(gmax - gmin, 1) * (W - PAD_L - PAD_R)

    def py(f):
        return H - PAD_B - (f - fmin) / (fmax - fmin) * (H - PAD_T - PAD_B)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
         f'height="{H}" style="background:#fafafa;font-family:monospace">']

    # Title
    s.append(f'<text x="{PAD_L}" y="30" font-size="16" fill="black" '
             f'font-weight="bold">{title}</text>')

    # Axes
    s.append(f'<line x1="{PAD_L}" y1="{H-PAD_B}" x2="{W-PAD_R}" '
             f'y2="{H-PAD_B}" stroke="black" stroke-width="1.5"/>')
    s.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" '
             f'y2="{H-PAD_B}" stroke="black" stroke-width="1.5"/>')

    # Grid + y labels
    for k in range(5):
        y = PAD_T + k * (H - PAD_T - PAD_B) // 4
        fv = fmax - k * (fmax - fmin) / 4
        s.append(f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" '
                 f'stroke="#eee" stroke-width="1"/>')
        s.append(f'<text x="{PAD_L-8}" y="{y+4}" text-anchor="end" '
                 f'font-size="10" fill="#666">{fv:+.4f}</text>')

    # Curve
    pts = ' '.join(f'{px(g):.1f},{py(f):.1f}' for g, f in zip(gens, fits))
    s.append(f'<polyline points="{pts}" fill="none" '
             f'stroke="#1a5276" stroke-width="2.2"/>')

    # Axis labels
    s.append(f'<text x="{W//2}" y="{H-18}" text-anchor="middle" '
             f'font-size="12" fill="#333">generations</text>')
    s.append(f'<text x="18" y="{H//2}" text-anchor="middle" '
             f'font-size="12" fill="#333" '
             f'transform="rotate(-90, 18, {H//2})">fitness</text>')

    s.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(s))


# ============================================================
#                Text Rendering
# ============================================================
def net_to_text(net, fit=None, gen=None):
    lines = []
    lines.append('=' * 60)
    if gen is not None:
        lines.append(f'NETWORK SNAPSHOT - gen {gen}')
    else:
        lines.append('NETWORK - final')
    lines.append('=' * 60)
    if fit is not None:
        lines.append(f'fitness: {fit:+.6f}')
    cross, rec, ws, wnz = net.stats()
    lines.append(f'total nodes: {net.total}   modules: {net.mods}')
    lines.append(f'active wires: {wnz}/{ws}   cross-module: {cross}   '
                 f'recursive: {rec}')
    lines.append(f'output module: {net.out_mod}   '
                 f'output nodes: {list(net.out_idx)}')
    lines.append('')

    lines.append('MODULE ASSIGNMENT:')
    s = 0
    for m, sz in enumerate(net.mods):
        mod_nodes = list(range(s, s + sz))
        tag = ' [OUTPUT]' if m == net.out_mod else ''
        lines.append(f'  M{m}{tag}: ' + ' '.join(f'n{i}' for i in mod_nodes))
        s += sz
    lines.append('')

    lines.append('INPUT WIRES (in_k -> node, weight):')
    any_input = False
    for i in range(net.total):
        ins = [(k, float(net.Win[i, k])) for k in range(net.n_in)
               if abs(net.Win[i, k]) > 0.05]
        if ins:
            any_input = True
            ws_str = ', '.join(f'in{k}({w:+.2f})' for k, w in ins)
            lines.append(f'  n{i:>2d} <- {ws_str}   '
                         f'bias={float(net.bias[i]):+.3f}')
    if not any_input:
        lines.append('  (none)')
    lines.append('')

    lines.append('INTERNAL WIRES (source -> target, weight):')
    any_int = False
    for i in range(net.total):
        for j in range(net.total):
            w = float(net.W[i, j])
            if abs(w) < 0.05:
                continue
            any_int = True
            src_m = net._mod_of(j)
            dst_m = net._mod_of(i)
            tag = ''
            if i == j:
                tag = ' [RECURSIVE]'
            elif src_m != dst_m:
                tag = f' [CROSS M{src_m}->M{dst_m}]'
            lines.append(f'  n{j:>2d} -> n{i:>2d}   w={w:+.3f}{tag}')
    if not any_int:
        lines.append('  (none)')
    lines.append('')

    lines.append('CROSS-MODULE SUMMARY:')
    cross_count = {}
    for i in range(net.total):
        for j in range(net.total):
            if abs(net.W[i, j]) < 0.05:
                continue
            ms = net._mod_of(j)
            md = net._mod_of(i)
            if ms != md:
                key = (ms, md)
                cross_count[key] = cross_count.get(key, 0) + 1
    if cross_count:
        for (ms, md), cnt in sorted(cross_count.items()):
            lines.append(f'  M{ms} -> M{md}: {cnt} wires')
    else:
        lines.append('  (no cross-module wires)')
    lines.append('')

    lines.append('RECURSIVE LOOPS:')
    rec_nodes = [i for i in range(net.total) if abs(net.W[i, i]) > 0.05]
    if rec_nodes:
        for i in rec_nodes:
            lines.append(f'  n{i} -> n{i}   w={float(net.W[i, i]):+.3f}')
    else:
        lines.append('  (none)')
    lines.append('')

    return '\n'.join(lines)


def curve_to_text(history, title=''):
    lines = [f'LEARNING CURVE - {title}', '=' * 50]
    gens = history['gen']
    fits = history['fit']
    if not gens:
        return '\n'.join(lines)
    n = len(gens)
    idx = ([int(i * (n - 1) / 14) for i in range(15)] if n > 15
           else list(range(n)))
    lines.append(f'{"gen":>6s}  {"fitness":>10s}  {"size":>5s}  '
                 f'{"cross":>6s}  {"rec":>4s}')
    lines.append('-' * 40)
    for i in idx:
        lines.append(f'{gens[i]:>6d}  {fits[i]:>+10.5f}  '
                     f'{history["size"][i]:>5d}  {history["cross"][i]:>6d}  '
                     f'{history["rec"][i]:>4d}')
    lines.append('')
    fmin, fmax = min(fits), max(fits)
    if fmax - fmin < 1e-6:
        fmax = fmin + 0.001
    H, W = 12, 50
    lines.append('ASCII PLOT (fitness over generations):')
    for row in range(H):
        fv = fmax - row * (fmax - fmin) / (H - 1)
        bar = ''
        for col in range(W):
            gi = int(col * (n - 1) / (W - 1))
            f_here = fits[gi]
            prev_fv = (fmax - (row - 1) * (fmax - fmin) / (H - 1)
                       if row > 0 else fmax + 1)
            next_fv = (fmax - (row + 1) * (fmax - fmin) / (H - 1)
                       if row < H - 1 else fmin - 1)
            bar += '*' if next_fv <= f_here <= prev_fv else ' '
        lines.append(f'  {fv:+.4f}  |{bar}|')
    lines.append(f'         +{"-" * W}+')
    lines.append(f'          gen 0{" " * (W - 10)}gen {gens[-1]}')
    return '\n'.join(lines)


def samples_to_csv(net, problem, path):
    X = problem['inputs']
    Y = problem['targets']
    steps = problem['steps']
    POWERS = problem['POWERS']
    DENOM = problem['DENOM']

    out_bits = net.forward_batch(X, steps)
    preds = net.predict_values(X, steps, POWERS, DENOM)

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'input_bits', 'target_value',
                    'output_bits', 'output_value', 'error'])
        for i in range(len(X)):
            inp = ''.join(str(int(b)) for b in X[i])
            out_b = ''.join(str(int(b)) for b in out_bits[i])
            w.writerow([i, inp, f'{Y[i]:.4f}', out_b,
                        f'{preds[i]:.4f}', f'{abs(preds[i] - Y[i]):.4f}'])


# ============================================================
#                Experiment Runner
# ============================================================
def run_problem(problem, idx_global):
    name = problem['name']
    print(f"\n{'=' * 60}")
    print(f"  [{idx_global}] {name}: {problem['desc']}")
    print(f"  n_in={problem['n_in']}  n_out={problem['n_out']}  "
          f"steps={problem['steps']}")
    print(f"  max_gens={problem['max_gens']}")
    print(f"{'=' * 60}")

    out_dir = OUT / name
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'snapshots').mkdir(exist_ok=True)

    net, fit, hist, snaps, ev_time = evolve(problem, problem['max_gens'])

    cross, rec, ws, wnz = net.stats()

    preds = net.predict_values(problem['inputs'], problem['steps'],
                               problem['POWERS'], problem['DENOM'])
    Y = problem['targets']

    if problem['kind'] == 'boolean':
        pred_bits = (preds >= 0.5).astype(int)
        correct = int((pred_bits == Y.astype(int)).sum())
        accuracy = correct / len(Y)
        rmse = math.sqrt(max(0, -fit))
        metric_str = f'acc={correct}/{len(Y)}'
    else:
        rmse = math.sqrt(max(0, -fit))
        accuracy = None
        metric_str = f'RMSE={rmse:.4f}'

    with open(out_dir / 'best.json', 'w') as f:
        json.dump(net.to_dict(), f, indent=2)

    with open(out_dir / 'best.txt', 'w') as f:
        f.write(net_to_text(net, fit=fit))

    draw_net_svg(net, out_dir / 'best.svg',
                 title=f'{name} - best (fit={fit:+.5f})', fit=fit)

    with open(out_dir / 'curve.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gen', 'fit', 'cross', 'rec', 'size', 'wires'])
        for i in range(len(hist['gen'])):
            w.writerow([hist['gen'][i], f"{hist['fit'][i]:.6f}",
                        hist['cross'][i], hist['rec'][i],
                        hist['size'][i], hist['wires'][i]])

    draw_curve_svg(hist, out_dir / 'curve.svg', title=name)
    with open(out_dir / 'curve.txt', 'w') as f:
        f.write(curve_to_text(hist, name))

    snap_lines = [f'SNAPSHOTS FOR {name}', '=' * 60]
    for gen, snap_net in sorted(snaps.items()):
        draw_net_svg(snap_net,
                     out_dir / 'snapshots' / f'gen_{gen:04d}.svg',
                     title=f'{name} @ gen {gen}')
        snap_lines.append('')
        snap_lines.append(net_to_text(snap_net, gen=gen))
    with open(out_dir / 'snapshots.txt', 'w') as f:
        f.write('\n'.join(snap_lines))

    samples_to_csv(net, problem, out_dir / 'samples.csv')

    print(f"  --> fit={fit:+.5f}  {metric_str}  "
          f"size={net.total}  cross={cross}  rec={rec}")
    print(f"  time: {ev_time:.1f}s")

    return {
        'name': name, 'desc': problem['desc'],
        'n_in': problem['n_in'], 'n_out': problem['n_out'],
        'steps': problem['steps'], 'kind': problem['kind'],
        'max_gens': problem['max_gens'],
        'best_fit': fit, 'rmse': rmse,
        'accuracy': accuracy,
        'size': net.total, 'cross': cross, 'rec': rec,
        'wires': wnz, 'mods': str(net.mods),
        'time': ev_time,
        'gens_run': hist['gen'][-1] if hist['gen'] else 0,
    }


def main():
    print('=' * 60)
    print('  Paper Experiment Framework - ModNet')
    print(f'  seed = {SEED}')
    print('=' * 60)

    problems = [
        make_xor2(),
        make_parity4(),
        make_parity6(),
        make_sin2pi(),
        make_sin4pi(),
        make_circle(),
    ]

    results = []
    t0 = time.time()
    for i, prob in enumerate(problems, 1):
        try:
            r = run_problem(prob, i)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {prob['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total = time.time() - t0

    with open(OUT / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'desc', 'kind', 'n_in', 'n_out', 'steps',
                    'max_gens', 'gens_run', 'best_fit', 'rmse', 'accuracy',
                    'size', 'cross', 'rec', 'wires', 'mods', 'time'])
        for r in results:
            w.writerow([r['name'], r['desc'], r['kind'],
                        r['n_in'], r['n_out'], r['steps'],
                        r['max_gens'], r['gens_run'],
                        f"{r['best_fit']:.6f}",
                        f"{r['rmse']:.6f}",
                        (f"{r['accuracy']:.4f}"
                         if r['accuracy'] is not None else ''),
                        r['size'], r['cross'], r['rec'],
                        r['wires'], r['mods'], f"{r['time']:.1f}"])

    lines = ['=' * 70,
             'PAPER EXPERIMENT SUMMARY',
             f'seed = {SEED}   total time = {total:.1f}s',
             '=' * 70, '']
    lines.append(f'{"problem":<10s} {"kind":<11s} {"fit":>10s} {"RMSE":>8s} '
                 f'{"acc":>6s} {"size":>5s} {"cross":>6s} {"rec":>4s} '
                 f'{"time":>7s}')
    lines.append('-' * 70)
    for r in results:
        acc = (f"{r['accuracy']*100:.0f}%"
               if r['accuracy'] is not None else '-')
        lines.append(f"{r['name']:<10s} {r['kind']:<11s} "
                     f"{r['best_fit']:>+10.5f} {r['rmse']:>8.4f} "
                     f"{acc:>6s} {r['size']:>5d} {r['cross']:>6d} "
                     f"{r['rec']:>4d} {r['time']:>6.1f}s")
    lines.append('')
    lines.append('DETAILS PER PROBLEM:')
    for r in results:
        lines.append('')
        lines.append(f'  {r["name"]} - {r["desc"]}')
        lines.append(f'    architecture: n_in={r["n_in"]}, '
                     f'n_out={r["n_out"]}, steps={r["steps"]}')
        lines.append(f'    modules: {r["mods"]}')
        lines.append(f'    nodes: {r["size"]}   '
                     f'cross-module wires: {r["cross"]}   '
                     f'recursive: {r["rec"]}')
        lines.append(f'    total active wires: {r["wires"]}')
        lines.append(f'    final fitness: {r["best_fit"]:+.6f}')
        lines.append(f'    RMSE: {r["rmse"]:.4f}')
        if r["accuracy"] is not None:
            lines.append(f'    accuracy: {r["accuracy"]*100:.1f}%')
        lines.append(f'    solved in {r["gens_run"]} generations, '
                     f'{r["time"]:.1f}s')
        lines.append(f'    files: results/{r["name"]}/')

    with open(OUT / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n{'=' * 60}")
    print(f"  All experiments complete.")
    print(f"  Total time: {total:.1f}s")
    print(f"  Output: {OUT}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
