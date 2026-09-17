"""
ModNet v017 — Lamarkian Evolution
==================================
هر چند نسل، نخبه‌ها یه گام گرادیان می‌گیرن.
اگه بهتر شدن، برمی‌گردن به جمعیت.
اگه بدتر شدن، برمی‌گردن به نسخه قبل.
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
PRUNE_EVERY = 15
PRUNE_TOL = -0.001

SIGMA_BASE = 0.3
SIGMA_MAX = 1.5
SIGMA_GROWTH = 1.25
PLATEAU_TRIGGER = 25
BIRTH_TRIGGER = 80
BIRTH_ELITES = 3
FRESH_RATE = 0.05
CROSSOVER_RATE = 0.30

# ★ Lamarkian ★
LAMARK_EVERY = 3       # هر چند نسل
LAMARK_ELITES = 30     # تعداد نخبه‌ها
LAMARK_ITERS = 8       # گام گرادیان هر نخبه
LAMARK_LR = 0.02       # یادگیری
LAMARK_T = 0.6         # sigmoid نرم (جلوگیری از vanishing)


def encode(x, bits=N_IN):
    v = int(round(x * (2**bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


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

    def migrate(self, other):
        new = self.clone()
        n_min = min(self.total, other.total)
        if n_min < 2:
            return new
        for _ in range(3):
            i = random.randrange(n_min)
            if random.random() < 0.5:
                new.W[i, :n_min] = other.W[i, :n_min]
                new.Win[i] = other.Win[i]
            else:
                new.bias[i] = other.bias[i]
        return new

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

    # ========================================================
    # گرادیان
    # ========================================================
    def gradient_step(self, data_in, data_tgt, lr, T, steps=N_STEPS):
        B = data_in.shape[0]
        const = data_in @ self.Win.T + self.bias
        Wt = self.W.T
        states = [np.zeros((B, self.total), dtype=np.float32)]
        for _ in range(steps):
            z = states[-1] @ Wt + const
            states.append(sigmoid(z / T))

        out = states[-1][:, self.out_idx]
        preds = (out @ POWERS) / DENOM
        dL_dpred = 2.0 * (preds - data_tgt) / B
        dL_dh = np.zeros((B, self.total), dtype=np.float32)
        for i, idx in enumerate(self.out_idx):
            dL_dh[:, idx] += dL_dpred * POWERS[i] / DENOM

        dW = np.zeros_like(self.W)
        dWin = np.zeros_like(self.Win)
        dbias = np.zeros_like(self.bias)

        for t in range(steps, 0, -1):
            h = states[t]
            dL_dz = dL_dh * h * (1.0 - h) / T
            h_prev = states[t-1]
            dW += dL_dz.T @ h_prev
            dWin += dL_dz.T @ data_in
            dbias += dL_dz.sum(axis=0)
            dL_dh = dL_dz @ self.W

        for g in (dW, dWin, dbias):
            np.clip(g, -2, 2, out=g)
        self.W -= lr * dW
        self.Win -= lr * dWin
        self.bias -= lr * dbias
        return float(-np.mean((preds - data_tgt) ** 2))

    def lamarkian_update(self, data_in, data_tgt, iters, lr, T, steps=N_STEPS):
        """
        چند گام گرادیان روی خودم می‌زنم.
        برمی‌گردونم: بهترین fit (با step) که دیدم.
        """
        best_net = self.clone()
        best_fit = self.fitness(data_in, data_tgt, steps)

        for _ in range(iters):
            self.gradient_step(data_in, data_tgt, lr, T, steps)
            fit = self.fitness(data_in, data_tgt, steps)
            if fit > best_fit + 1e-7:
                best_fit = fit
                best_net = self.clone()

        # کپی بهترین به خودم
        self.W = best_net.W.copy()
        self.Win = best_net.Win.copy()
        self.bias = best_net.bias.copy()
        return best_fit

    def stats(self):
        Wnz = np.nonzero(self.W)
        cross = sum(1 for i, j in zip(Wnz[0], Wnz[1])
                    if self._mod_map[i] != self._mod_map[j])
        rec = sum(1 for i, j in zip(Wnz[0], Wnz[1]) if i == j)
        return cross, rec, int(self.W.size), int((self.W != 0).sum())


# ============================================================
def _eval_population(pop, data_in, data_tgt, steps=N_STEPS):
    results = [0.0] * len(pop)
    by_size = {}
    for i, net in enumerate(pop):
        by_size.setdefault(net.total, []).append(i)

    B = data_in.shape[0]
    for size, indices in by_size.items():
        G = len(indices)
        N = size

        W_s = np.stack([pop[i].W for i in indices])
        Win_s = np.stack([pop[i].Win for i in indices])
        b_s = np.stack([pop[i].bias for i in indices])
        oi_s = np.stack([pop[i].out_idx for i in indices])

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

        for k, i in enumerate(indices):
            results[i] = float(fits[k])
    return results


# ============================================================
def evolve(data_in, data_tgt, pop_size=200, gens=1500, verbose=True,
           report_every=200):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT) for _ in range(pop_size)]

    if verbose:
        print(f"  حالت: Lamarkian Evolution")
        print(f"  جمعیت: {pop_size}   نسل: {gens}   N_STEPS={N_STEPS}")
        print(f"  Lamark: هر {LAMARK_EVERY} نسل، {LAMARK_ELITES} نخبه، "
              f"{LAMARK_ITERS} گام × lr={LAMARK_LR}، T={LAMARK_T}")
        print()

    t_eval = 0.0
    t_birth = 0.0
    t_prune = 0.0
    t_repro = 0.0
    t_lamark = 0.0
    n_lamark_accepted = 0
    n_lamark_total = 0

    best_fit = -1e9
    best_net = None
    t_start = time.time()
    plateau = 0
    sigma = SIGMA_BASE
    elite_size = pop_size // 5

    for gen in range(gens):
        # ── ارزیابی ──
        ts = time.time()
        fits = _eval_population(pop, data_in, data_tgt, N_STEPS)
        t_eval += time.time() - ts

        fits_arr = np.array(fits, dtype=np.float64)
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

        # ── Lamarkian ──
        if gen > 0 and gen % LAMARK_EVERY == 0:
            ts = time.time()
            accepted = 0
            for k in order[:LAMARK_ELITES]:
                k = int(k)
                net = pop[k]
                old_fit = float(fits_arr[k])
                new_fit = net.lamarkian_update(
                    data_in, data_tgt,
                    iters=LAMARK_ITERS,
                    lr=LAMARK_LR,
                    T=LAMARK_T,
                    steps=N_STEPS
                )
                n_lamark_total += 1
                if new_fit > old_fit + 1e-7:
                    fits_arr[k] = new_fit
                    accepted += 1
                    n_lamark_accepted += 1
            t_lamark += time.time() - ts

            # آپدیت best
            new_top_i = int(np.argmax(fits_arr))
            new_top = float(fits_arr[new_top_i])
            if new_top > best_fit + 1e-5:
                best_fit = new_top
                best_net = pop[new_top_i].clone()
                improved = True

            if verbose and accepted > 0:
                print(f"  gen {gen:4d}   🧬 Lamark: {accepted}/{LAMARK_ELITES} پذیرفته   "
                      f"best={best_fit:+.5f}")

        # ── تولد ──
        if gen > 0 and gen % BIRTH_TRIGGER == 0:
            ts = time.time()
            born = 0
            for k in order[:BIRTH_ELITES]:
                k = int(k)
                child = pop[k].birth()
                if child is not None:
                    pop[k] = child
                    fits_arr[k] = child.fitness(data_in, data_tgt, N_STEPS)
                    born += 1
            t_birth += time.time() - ts
            if verbose and born > 0:
                print(f"  gen {gen:4d}   🐣 {born} نورون   size={best_net.total}")

        # ── هرس ──
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
        # دوباره مرتب کن (چون Lamarkian ممکنه تغییر داده باشه)
        order = np.argsort(fits_arr)[::-1]
        elite = [pop[int(i)] for i in order[:elite_size]]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            r = random.random()
            if r < CROSSOVER_RATE and len(elite) >= 2:
                p1, p2 = random.sample(elite, 2)
                child = p1.crossover(p2)
                if child is None:
                    child = p1.mutate(random.randint(1, 3), sigma=sigma)
                else:
                    child = child.mutate(1, sigma=sigma * 0.5)
            elif r < 1.0 - FRESH_RATE:
                parent = random.choice(elite)
                child = parent.mutate(random.randint(1, 4), sigma=sigma)
            else:
                child = ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT)
            new_pop.append(child)
        pop = new_pop
        t_repro += time.time() - ts

        if verbose and gen > 0 and gen % report_every == 0:
            elapsed = time.time() - t_start
            per_gen = elapsed / (gen + 1)
            print(f"  ⏱  gen={gen}  کل={elapsed:.1f}s  ({per_gen*1000:.0f}ms/gen)")
            print(f"     eval={100*t_eval/elapsed:.0f}%  "
                  f"lamark={100*t_lamark/elapsed:.0f}% "
                  f"({n_lamark_accepted}/{n_lamark_total} پذیرفته)  "
                  f"prune={100*t_prune/elapsed:.0f}%  "
                  f"repro={100*t_repro/elapsed:.0f}%")

    total_time = time.time() - t_start

    if verbose:
        print(f"\n  ═══ گزارش نهایی ═══")
        print(f"  کل: {total_time:.1f}s")
        print(f"    eval:   {t_eval:.1f}s ({100*t_eval/total_time:.0f}%)")
        print(f"    lamark: {t_lamark:.1f}s ({100*t_lamark/total_time:.1f}%)  "
              f"({n_lamark_accepted}/{n_lamark_total} پذیرفته = "
              f"{100*n_lamark_accepted/max(n_lamark_total,1):.0f}%)")
        print(f"    prune:  {t_prune:.1f}s ({100*t_prune/total_time:.1f}%)")
        print(f"    repro:  {t_repro:.1f}s ({100*t_repro/total_time:.1f}%)")

    return best_net, best_fit, total_time


def main():
    print("=" * 68)
    print("  ModNet v017 — Lamarkian Evolution")
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
