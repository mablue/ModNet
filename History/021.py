"""
ModNet v021 — سرعت نهایی
==========================
۱. mutate درجا (بدون clone اولیه)
۲. batch mutation با numpy
۳. crossover سریع‌تر
۴. cache (از v020)
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np
import random, math, time

N_IN = 4
N_OUT = 8
N_STEPS = 4
MODULE_SIZES = [6, 6, 6, 8]
OUT_MODULE = 3
POWERS = np.array([1,2,4,8,16,32,64,128], dtype=np.float32)
DENOM = 255.0

MIN_MODULE_SIZE = 2
MAX_MODULE_SIZE = 15
MAX_TOTAL = 60

W_THRESH = 0.04
ACT_LO = 0.02
ACT_HI = 0.98
PRUNE_EVERY = 20
PRUNE_TOL = -0.001

SIGMA_BASE = 0.3
SIGMA_MAX = 1.5
SIGMA_GROWTH = 1.25
PLATEAU_TRIGGER = 25
BIRTH_TRIGGER = 100
BIRTH_ELITES = 3
FRESH_RATE = 0.05
CROSSOVER_RATE = 0.30


def encode(x, bits=N_IN):
    v = int(round(x * (2**bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]


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
            for i in range(s, s+sz):
                self._mod_map[i] = m
            s += sz
        out_all = [i for i in range(self.total) if self._mod_map[i] == self.out_mod]
        self.out_idx = np.array(out_all[:self.out_size], dtype=np.int32)
        if len(self.out_idx) < self.out_size:
            extra = [i for i in range(self.total)
                     if self._mod_map[i] != self.out_mod][:self.out_size - len(self.out_idx)]
            self.out_idx = np.array(list(self.out_idx) + extra, dtype=np.int32)

    def _mod_of(self, i): return int(self._mod_map[i])

    def _init(self):
        for i in range(self.total):
            for _ in range(random.randint(1, 3)):
                if self._mod_of(i) == self.out_mod or random.random() > 0.3:
                    self.W[i, random.randrange(self.total)] += random.gauss(0, 1)
                else:
                    self.Win[i, random.randrange(self.n_in)] += random.gauss(0, 1)
            self.bias[i] = random.gauss(0, 1)

    def forward_batch(self, inputs_batch, steps=N_STEPS):
        B = inputs_batch.shape[0]
        state = np.zeros((B, self.total), dtype=np.float32)
        const = inputs_batch @ self.Win.T + self.bias
        Wt = self.W.T
        for _ in range(steps):
            state = ((state @ Wt + const) >= 0).astype(np.float32)
        return state[:, self.out_idx]

    def activity(self, inputs_batch, steps=N_STEPS):
        B = inputs_batch.shape[0]
        state = np.zeros((B, self.total), dtype=np.float32)
        const = inputs_batch @ self.Win.T + self.bias
        Wt = self.W.T
        total_act = np.zeros(self.total, dtype=np.float64)
        for _ in range(steps):
            state = ((state @ Wt + const) >= 0).astype(np.float32)
            total_act += state.mean(axis=0)
        return total_act / steps

    def fitness(self, data_in, data_tgt, steps=N_STEPS):
        out = self.forward_batch(data_in, steps)
        preds = (out @ POWERS) / DENOM
        return float(-np.mean((preds - data_tgt) ** 2))

    def clone(self):
        """کلون سبک — فقط W و Win و bias کپی می‌شن (چون تغییرپذیرن)"""
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = self.mods          # ← reference، نه copy
        n.out_mod = self.out_mod
        n.out_size = self.out_size
        n.total = self.total
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        n._mod_map = self._mod_map  # ← reference
        n.out_idx = self.out_idx    # ← reference
        return n

    # ========================================================
    # ★ ترفند ۱ + ۲: mutate درجا با numpy ★
    # ========================================================
    def mutate_inplace(self, rate=2, sigma=0.5):
        """clone + جهش در یک عملیات"""
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = self.mods
        n.out_mod = self.out_mod
        n.out_size = self.out_size
        n.total = self.total
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        n._mod_map = self._mod_map
        n.out_idx = self.out_idx

        T = n.total
        # ★ همه‌ی جهش‌ها با numpy ★
        # انتخاب تصادفی نوع جهش برای هر rate
        r = np.random.random(rate)
        idx = np.random.randint(0, T, size=rate)
        jdx = np.random.randint(0, T, size=rate)
        jdx_in = np.random.randint(0, n.n_in, size=rate)
        deltas = np.random.normal(0, sigma, size=rate)
        # برای "حذف یال" — فقط اگه قبلاً یال داشتیم
        for k in range(rate):
            rk = r[k]
            i = int(idx[k])
            if rk < 0.40:
                # جهش وزن W
                n.W[i, jdx[k]] += deltas[k]
            elif rk < 0.60:
                # جهش bias
                n.bias[i] += deltas[k]
            elif rk < 0.80:
                # جهش وزن Win
                n.Win[i, jdx_in[k]] += deltas[k]
            elif rk < 0.92:
                # حذف یه یال تصادفی
                row = n.W[i]
                nz = np.nonzero(row)[0]
                if len(nz) > 1:
                    n.W[i, int(nz[np.random.randint(len(nz))])] = 0
            else:
                # افزودن یال
                n.W[i, jdx[k]] = np.random.normal(0, 1)
        return n

    def birth(self):
        if self.total >= MAX_TOTAL:
            return None
        candidates = [m for m in range(len(self.mods)) if self.mods[m] < MAX_MODULE_SIZE]
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
        insert_pos = sum(self.mods[:chosen_mod+1])
        N = self.total
        W_new = np.zeros((N+1, N+1), dtype=np.float32)
        Win_new = np.zeros((N+1, self.n_in), dtype=np.float32)
        bias_new = np.zeros(N+1, dtype=np.float32)
        for i in range(N):
            new_i = i if i < insert_pos else i + 1
            bias_new[new_i] = self.bias[i]
            Win_new[new_i] = self.Win[i]
            for j in range(N):
                new_j = j if j < insert_pos else j + 1
                W_new[new_i, new_j] = self.W[i, j]
        for _ in range(random.randint(1, 3)):
            src = random.randrange(N+1)
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

    # ========================================================
    # ★ ترفند ۳: crossover سریع ★
    # ========================================================
    def crossover_fast(self, other):
        if self.total != other.total or self.mods != other.mods:
            return None
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = self.mods
        n.out_mod = self.out_mod
        n.out_size = self.out_size
        n.total = self.total
        n._mod_map = self._mod_map
        n.out_idx = self.out_idx
        # ★ بردار تصمیم: ۵۰٪ از مادر، ۵۰٪ از پدر ★
        T = self.total
        mask = np.random.random(T) < 0.5     # (T,)
        # W: هر سطر از یکی
        n.W = np.where(mask[:, None], self.W, other.W).astype(np.float32)
        # Win: هر سطر از یکی
        n.Win = np.where(mask[:, None], self.Win, other.Win).astype(np.float32)
        # bias: هر درایه از یکی
        n.bias = np.where(mask, self.bias, other.bias).astype(np.float32)
        return n

    def _prune_inplace(self, data_in, min_size=MIN_MODULE_SIZE):
        act = self.activity(data_in)
        self.W[np.abs(self.W) < W_THRESH] = 0
        self.Win[np.abs(self.Win) < W_THRESH] = 0
        alive = np.ones(self.total, dtype=bool)
        alive &= ~((act < ACT_LO) | (act > ACT_HI))
        in_from_W = np.abs(self.W).sum(axis=0)
        in_from_in = np.abs(self.Win).sum(axis=1)
        out_to = np.abs(self.W).sum(axis=1)
        alive &= ((in_from_W + in_from_in) > 0) | (out_to > 0)
        per_mod_keep = []
        s = 0
        for m, sz in enumerate(self.mods):
            idx = list(range(s, s+sz))
            keep_idx = [i for i in idx if alive[i]]
            if len(keep_idx) < min_size:
                sorted_idx = sorted(idx, key=lambda i: -abs(act[i] - 0.5))
                keep_idx = sorted_idx[:min_size]
            per_mod_keep.append(sorted(keep_idx))
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

    def try_prune(self, data_in, data_tgt, old_fit=None, steps=N_STEPS):
        if old_fit is None:
            old_fit = self.fitness(data_in, data_tgt, steps)
        candidate = self.clone()
        removed = candidate._prune_inplace(data_in)
        if removed == 0:
            return None
        new_fit = candidate.fitness(data_in, data_tgt, steps)
        if new_fit >= old_fit + PRUNE_TOL:
            return candidate, removed, new_fit
        return None

    def stats(self):
        Wnz = np.nonzero(self.W)
        cross = sum(1 for i, j in zip(Wnz[0], Wnz[1])
                    if self._mod_map[i] != self._mod_map[j])
        rec = sum(1 for i, j in zip(Wnz[0], Wnz[1]) if i == j)
        return cross, rec, int(self.W.size), int((self.W != 0).sum())


# ============================================================
def _eval_population(pop, data_in, data_tgt, steps=N_STEPS, indices=None):
    if indices is None:
        indices = list(range(len(pop)))
    n = len(indices)
    results = [0.0] * n
    if n == 0:
        return results
    by_size = {}
    for k, i in enumerate(indices):
        by_size.setdefault(pop[i].total, []).append(k)
    B = data_in.shape[0]
    for size, local_idx in by_size.items():
        G = len(local_idx)
        N = size
        W_s = np.stack([pop[indices[k]].W for k in local_idx])
        Win_s = np.stack([pop[indices[k]].Win for k in local_idx])
        b_s = np.stack([pop[indices[k]].bias for k in local_idx])
        oi_s = np.stack([pop[indices[k]].out_idx for k in local_idx])
        const = np.matmul(Win_s, data_in.T) + b_s[:, :, None]
        const = np.transpose(const, (0, 2, 1))
        Wt = np.transpose(W_s, (0, 2, 1))
        state = np.zeros((G, B, N), dtype=np.float32)
        for _ in range(steps):
            z = np.matmul(state, Wt) + const
            state = (z >= 0).astype(np.float32)
        idx_b = np.broadcast_to(oi_s[:, None, :], (G, B, oi_s.shape[1]))
        out = np.take_along_axis(state, idx_b, axis=2)
        preds = (out @ POWERS) / DENOM
        diffs = (preds - data_tgt[None, :]) ** 2
        fits = -np.mean(diffs, axis=1)
        for k, val in zip(local_idx, fits):
            results[k] = float(val)
    return results


# ============================================================
def evolve(data_in, data_tgt, pop_size=300, gens=2000, verbose=True,
           report_every=200):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT) for _ in range(pop_size)]
    fits_cache = [None] * pop_size

    if verbose:
        print(f"  حالت: v021 (3 ترفند + cache), N_STEPS={N_STEPS}")
        print(f"  جمعیت: {pop_size}   نسل: {gens}")
        print()

    t_eval = 0.0; t_birth = 0.0; t_prune = 0.0; t_repro = 0.0
    n_eval_calls = 0
    n_eval_skipped = 0

    best_fit = -1e9
    best_net = None
    t_start = time.time()
    plateau = 0
    sigma = SIGMA_BASE
    elite_size = pop_size // 5

    for gen in range(gens):
        ts = time.time()
        to_eval = [i for i in range(pop_size) if fits_cache[i] is None]
        n_eval_calls += len(to_eval)
        n_eval_skipped += pop_size - len(to_eval)
        if to_eval:
            new_fits = _eval_population(pop, data_in, data_tgt, N_STEPS,
                                         indices=to_eval)
            for k, i in enumerate(to_eval):
                fits_cache[i] = new_fits[k]
        t_eval += time.time() - ts

        fits_arr = np.array(fits_cache, dtype=np.float64)
        order = np.argsort(fits_arr)[::-1]
        top_fit = float(fits_arr[order[0]])

        improved = False
        if top_fit > best_fit + 1e-5:
            best_fit = top_fit
            best_net = pop[int(order[0])].clone()
            plateau = 0
            sigma = SIGMA_BASE
            improved = True
        else:
            plateau += 1
            if plateau > PLATEAU_TRIGGER:
                sigma = min(SIGMA_MAX, sigma * SIGMA_GROWTH)
                plateau = 0

        if gen > 0 and gen % BIRTH_TRIGGER == 0:
            ts = time.time()
            born = 0
            for k in order[:BIRTH_ELITES]:
                k = int(k)
                child = pop[k].birth()
                if child is not None:
                    pop[k] = child
                    fits_cache[k] = None
                    born += 1
            t_birth += time.time() - ts
            if verbose and born > 0:
                print(f"  gen {gen:4d}   🐣 {born} نورون   size={best_net.total}")

        if gen > 0 and gen % PRUNE_EVERY == 0:
            ts = time.time()
            total_removed = 0
            accepted = 0
            for k in order[:elite_size]:
                k = int(k)
                result = pop[k].try_prune(data_in, data_tgt,
                                          old_fit=float(fits_arr[k]))
                if result is not None:
                    cand, removed, nf = result
                    pop[k] = cand
                    fits_cache[k] = nf
                    fits_arr[k] = nf
                    total_removed += removed
                    accepted += 1
            t_prune += time.time() - ts
            new_order = np.argsort(fits_arr)[::-1]
            top_i = int(new_order[0])
            new_top = float(fits_arr[top_i])
            if new_top > best_fit + 1e-5:
                best_fit = new_top
                best_net = pop[top_i].clone()
                improved = True
            elif (abs(new_top - best_fit) <= 1e-5
                  and pop[top_i].total < best_net.total):
                best_net = pop[top_i].clone()
            if verbose and accepted > 0:
                print(f"  gen {gen:4d}   ✂ {accepted} هرس، "
                      f"-{total_removed} نورون   size={best_net.total}")

        if verbose and improved:
            c, r, W_tot, W_nz = best_net.stats()
            print(f"  gen {gen:4d}   fit={best_fit:+.5f}   "
                  f"cross={c:3d} rec={r:3d}   "
                  f"size={best_net.total}   σ={sigma:.2f}")

        if best_fit > -0.0005:
            break

        # ── تولید نسل بعد ──
        ts = time.time()
        order = np.argsort(fits_arr)[::-1]
        elite_indices = [int(i) for i in order[:elite_size]]
        new_pop = [pop[i] for i in elite_indices]
        new_cache = [fits_cache[i] for i in elite_indices]
        elite_nets = new_pop

        while len(new_pop) < pop_size:
            r = random.random()
            if r < CROSSOVER_RATE and len(elite_nets) >= 2:
                p1, p2 = random.sample(elite_nets, 2)
                child = p1.crossover_fast(p2)       # ★ ترفند ۳
                if child is None:
                    child = p1.mutate_inplace(random.randint(1, 3), sigma=sigma)
                else:
                    child = child.mutate_inplace(1, sigma=sigma * 0.5)
            elif r < 1.0 - FRESH_RATE:
                parent = random.choice(elite_nets)
                child = parent.mutate_inplace(random.randint(1, 4), sigma=sigma)  # ★ ترفند ۱+۲
            else:
                child = ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT)
            new_pop.append(child)
            new_cache.append(None)
        pop = new_pop
        fits_cache = new_cache
        t_repro += time.time() - ts

        if verbose and gen > 0 and gen % report_every == 0:
            elapsed = time.time() - t_start
            per_gen = elapsed / (gen + 1)
            hit = 100 * n_eval_skipped / max(n_eval_calls + n_eval_skipped, 1)
            print(f"  ⏱  gen={gen}  کل={elapsed:.1f}s  ({per_gen*1000:.0f}ms/gen)")
            print(f"     eval={100*t_eval/elapsed:.0f}%  "
                  f"cache-hit={hit:.0f}%  "
                  f"prune={100*t_prune/elapsed:.0f}%  "
                  f"repro={100*t_repro/elapsed:.0f}%")

    total_time = time.time() - t_start
    if verbose:
        print(f"\n  ═══ گزارش نهایی ═══")
        print(f"  کل: {total_time:.1f}s")
        print(f"    eval:  {t_eval:.1f}s ({100*t_eval/total_time:.0f}%)  "
              f"cache-hit={100*n_eval_skipped/max(n_eval_calls+n_eval_skipped,1):.0f}%")
        print(f"    prune: {t_prune:.1f}s ({100*t_prune/total_time:.1f}%)")
        print(f"    repro: {t_repro:.1f}s ({100*t_repro/total_time:.1f}%)")

    return best_net, best_fit, total_time


def main():
    print("=" * 68)
    print("  ModNet v021 — سه ترفند + cache")
    print("=" * 68)
    N = 32
    xs = np.linspace(0, 1, N, dtype=np.float32)
    inputs = np.array([encode(float(x)) for x in xs], dtype=np.float32)
    targets = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)
    print(f"  داده: {N} نقطه   ورودی: {N_IN} بیت   خروجی: {N_OUT} بیت")
    print()

    net, fit, dt = evolve(inputs, targets)
    c, r, W_tot, W_nz = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0
    print(f"\n  نتیجه: fit={fit:+.5f}  RMSE={rmse:.4f}  زمان={dt:.1f}s")
    print(f"  پیمانه‌ها: {net.mods}  کل: {net.total}")
    print(f"  cross={c}  rec={r}  wires={W_nz}/{W_tot}")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for i in range(N):
        out = net.forward_batch(inputs[i:i+1], N_STEPS)[0]
        yp = float(out @ POWERS) / DENOM
        print(f"  {xs[i]:6.3f}  {targets[i]:7.3f}  {yp:10.3f}  {abs(yp-targets[i]):7.3f}")


if __name__ == '__main__':
    main()
