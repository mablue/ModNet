"""
ModNet v004 — نورون خالص + هرس مغزی
====================================
هرس:
  1. نورون‌های مرده (فعالیت ثابت) حذف
  2. وزن‌های نزدیک صفر صفر می‌شن
  3. نورون‌های منزوی حذف
  4. Compact کردن ماتریس‌ها
"""
import numpy as np
import random, math, time
from multiprocessing import Pool, cpu_count

N_IN = 4
N_OUT = 8
N_STEPS = 8
MODULE_SIZES = [6, 6, 6, 8]
OUT_MODULE = 3
POWERS = np.array([1,2,4,8,16,32,64,128], dtype=np.float32)
DENOM = 255.0
MIN_MODULE_SIZE = 3
W_THRESH = 0.03
ACT_LO = 0.03
ACT_HI = 0.97
PRUNE_EVERY = 25


def encode(x, bits=N_IN):
    v = int(round(x * (2**bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]


# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module, out_size):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.total = sum(module_sizes)
        self.out_size = out_size
        self._alloc()
        self._init()

    def _alloc(self):
        self.total = sum(self.mods)
        self.W = np.zeros((self.total, self.total), dtype=np.float32)
        self.Win = np.zeros((self.total, self.n_in), dtype=np.float32)
        self.bias = np.zeros(self.total, dtype=np.float32)
        self._mod_map = np.zeros(self.total, dtype=np.int32)
        s = 0
        for m, sz in enumerate(self.mods):
            for i in range(s, s+sz):
                self._mod_map[i] = m
            s += sz
        self.out_idx = np.array(
            [i for i in range(self.total) if self._mod_map[i] == self.out_mod][:self.out_size],
            dtype=np.int32
        )

    def _mod_of(self, i): return int(self._mod_map[i])

    def _init(self):
        for i in range(self.total):
            n_conn = random.randint(1, 3)
            for _ in range(n_conn):
                if self._mod_of(i) == self.out_mod or random.random() > 0.3:
                    j = random.randrange(self.total)
                    self.W[i, j] += random.gauss(0, 1)
                else:
                    j = random.randrange(self.n_in)
                    self.Win[i, j] += random.gauss(0, 1)
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
        """میانگین فعال بودن هر نورون در طول گام‌ها"""
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
            if r < 0.35:
                j = random.randrange(new.total)
                new.W[i, j] += random.gauss(0, sigma) if new.W[i, j] != 0 else random.gauss(0, 1)
            elif r < 0.55:
                new.bias[i] += random.gauss(0, sigma)
            elif r < 0.75:
                j = random.randrange(new.n_in)
                new.Win[i, j] += random.gauss(0, sigma) if new.Win[i, j] != 0 else random.gauss(0, 1)
            elif r < 0.90:
                nz = np.nonzero(new.W[i])[0]
                if len(nz) > 1:
                    new.W[i, int(random.choice(nz))] = 0
            else:
                j = random.randrange(new.total)
                new.W[i, j] = random.gauss(0, 1)
        return new

    def migrate(self, other):
        new = self.clone()
        for _ in range(3):
            i = random.randrange(new.total)
            if random.random() < 0.5:
                new.W[i] = other.W[i].copy()
            else:
                new.bias[i] = other.bias[i]
        return new

    # ========================================================
    #  هرس — قلب نسخه ۰۰۴
    # ========================================================
    def prune(self, data_in, min_size=MIN_MODULE_SIZE):
        """هرس نورون‌ها و اتصالات + Compact"""
        act = self.activity(data_in)

        # ۱. صفر کردن وزن‌های کوچیک
        self.W[np.abs(self.W) < W_THRESH] = 0
        self.Win[np.abs(self.Win) < W_THRESH] = 0

        # ۲. کدوم نورون‌ها زنده می‌مونن؟
        alive_mask = np.ones(self.total, dtype=bool)

        # نورون‌های مرده (فعالیت ثابت)
        dead = (act < ACT_LO) | (act > ACT_HI)
        alive_mask &= ~dead

        # نورون‌های منزوی (بدون ورودی و بدون خروجی)
        has_in = (np.abs(self.W).sum(axis=0) + np.abs(self.Win).sum(axis=0)) > 0
        has_out = np.abs(self.W).sum(axis=1) > 0
        isolated = ~(has_in | has_out)
        alive_mask &= ~isolated

        # نورون‌های غیرقابل دسترس (نه از خروجی قابل دسترسی، نه به خروجی می‌رسن)
        reachable = self._compute_reachability()
        alive_mask &= reachable

        # ۳. هر پیمانه حداقل min_size نورون نگه دار
        per_module = []
        s = 0
        for m, sz in enumerate(self.mods):
            idx = list(range(s, s + sz))
            alive_idx = [i for i in idx if alive_mask[i]]
            if len(alive_idx) < min_size:
                # مرتب بر اساس فعالیت و نگه‌داشتن فعال‌ترین‌ها
                sorted_idx = sorted(idx, key=lambda i: abs(act[i] - 0.5), reverse=True)
                alive_idx = sorted_idx[:min_size]
            per_module.append(sorted(alive_idx))
            s += sz

        # ۴. مطمئن شو پیمانه خروجی حداقل N_OUT نورون داره
        out_mod_alive = per_module[self.out_mod]
        if len(out_mod_alive) < self.out_size:
            # همه نورون‌های پیمانه خروجی رو نگه دار
            s = sum(self.mods[:self.out_mod])
            per_module[self.out_mod] = list(range(s, s + self.mods[self.out_mod]))

        # ۵. Compact: ماتریس‌های جدید
        keep = []
        for m_idx in per_module:
            keep.extend(m_idx)
        keep = np.array(keep, dtype=np.int32)

        new_mods = [len(m) for m in per_module]
        new_total = len(keep)

        new_W = np.zeros((new_total, new_total), dtype=np.float32)
        new_Win = np.zeros((new_total, self.n_in), dtype=np.float32)
        new_bias = np.zeros(new_total, dtype=np.float32)

        # نقشه: ایندکس قدیمی → جدید
        old_to_new = {int(o): i for i, o in enumerate(keep)}

        for i_new, i_old in enumerate(keep):
            new_bias[i_new] = self.bias[i_old]
            new_Win[i_new] = self.Win[i_old]
            for j_old in range(self.total):
                if j_old in old_to_new:
                    new_W[i_new, old_to_new[j_old]] = self.W[i_old, j_old]

        # ۶. جایگزینی
        self.mods = new_mods
        self.W = new_W
        self.Win = new_Win
        self.bias = new_bias
        self._alloc_no_init()

        removed = self.total - new_total
        return removed

    def _alloc_no_init(self):
        """مثل _alloc ولی بدون تصادفی‌سازی مجدد"""
        old_mods = list(self.mods)
        self.total = sum(old_mods)
        self._mod_map = np.zeros(self.total, dtype=np.int32)
        s = 0
        for m, sz in enumerate(old_mods):
            for i in range(s, s + sz):
                self._mod_map[i] = m
            s += sz
        self.out_idx = np.array(
            [i for i in range(self.total) if self._mod_map[i] == self.out_mod][:self.out_size],
            dtype=np.int32
        )

    def _compute_reachability(self):
        """نورون‌هایی که هم از ورودی می‌گیرن هم به خروجی می‌رسن"""
        total = self.total
        # forward reach: چه نورون‌هایی از ورودی‌ها سیگنال می‌گیرن
        fwd = np.zeros(total, dtype=bool)
        # یه بار از ورودی شروع کن
        direct_in = (np.abs(self.Win).sum(axis=1) > 0)
        fwd |= direct_in
        # BFS ساده روی گراف
        for _ in range(total):
            new_fwd = fwd.copy()
            for i in range(total):
                if fwd[i]:
                    out_edges = np.nonzero(self.W[:, i])[0]
                    new_fwd[out_edges] = True
            if np.array_equal(new_fwd, fwd):
                break
            fwd = new_fwd

        # backward reach: چه نورون‌هایی به خروجی می‌رسن
        bwd = np.zeros(total, dtype=bool)
        bwd[self.out_idx] = True
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

    def stats(self):
        Wnz = np.nonzero(self.W)
        cross = sum(1 for i, j in zip(Wnz[0], Wnz[1])
                    if self._mod_map[i] != self._mod_map[j])
        rec = sum(1 for i, j in zip(Wnz[0], Wnz[1]) if i == j)
        return cross, rec, int(self.W.size), int((self.W != 0).sum())


# ============================================================
_DATA_IN = None
_DATA_TGT = None

def _init_worker(din, dtgt):
    global _DATA_IN, _DATA_TGT
    _DATA_IN = din
    _DATA_TGT = dtgt

def _fitness_worker(net):
    return net.fitness(_DATA_IN, _DATA_TGT, N_STEPS)


# ============================================================
def evolve(data_in, data_tgt, pop_size=200, gens=1000, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT) for _ in range(pop_size)]

    n_proc = max(1, cpu_count())
    if verbose:
        print(f"  هسته‌ها: {n_proc}   جمعیت: {pop_size}   نسل: {gens}")
        print(f"  هرس هر {PRUNE_EVERY} نسل   min_module={MIN_MODULE_SIZE}")
        print()

    best_fit = -1e9
    best_net = None
    t0 = time.time()
    plateau = 0
    chunk = max(1, pop_size // (n_proc * 4))

    with Pool(n_proc, initializer=_init_worker,
              initargs=(data_in, data_tgt)) as pool:
        for gen in range(gens):
            fits = pool.map(_fitness_worker, pop, chunksize=chunk)
            best_idx = int(np.argmax(fits))
            top_fit = fits[best_idx]

            improved = False
            if top_fit > best_fit + 1e-5:
                best_fit = top_fit
                best_net = pop[best_idx].clone()
                plateau = 0
                improved = True
            else:
                plateau += 1

            # ---- هرس دوره‌ای ----
            if gen > 0 and gen % PRUNE_EVERY == 0:
                sizes_before = [n.total for n in pop[:20]]
                # فقط بهترین‌ها رو هرس کن (سریع‌تر)
                idx_sort = np.argsort(fits)[::-1]
                elite_n = pop_size // 5
                for k in idx_sort[:elite_n]:
                    n = pop[int(k)]
                    n.prune(data_in, MIN_MODULE_SIZE)
                sizes_after = [n.total for n in pop[:elite_n]]
                saved = sum(sizes_before[:elite_n]) - sum(sizes_after)
                if verbose and improved:
                    c, r, W_total, W_nz = best_net.stats()
                    print(f"  gen {gen:4d}   fit={best_fit:+.5f}   "
                          f"cross={c:3d} rec={r:3d}   "
                          f"wires={W_nz}/{W_total}   ✂ -{saved} نورون")

            if verbose and improved:
                c, r, W_total, W_nz = best_net.stats()
                print(f"  gen {gen:4d}   fit={best_fit:+.5f}   "
                      f"cross={c:3d} rec={r:3d}   "
                      f"wires={W_nz}/{W_total}   size={best_net.total}")

            if plateau > 120:
                if verbose:
                    print(f"  gen {gen:4d}   ⚡ restart")
                keep = pop_size // 5
                idx = np.argsort(fits)[::-1][:keep]
                elite = [pop[int(i)] for i in idx]
                pop = list(elite) + [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT)
                                     for _ in range(pop_size - keep)]
                plateau = 0
                continue

            if best_fit > -0.0005: break

            idx = np.argsort(fits)[::-1]
            elite_n = pop_size // 5
            elite = [pop[int(i)] for i in idx[:elite_n]]

            new_pop = list(elite)
            while len(new_pop) < pop_size:
                parent = random.choice(elite)
                child = parent.mutate(random.randint(1, 4),
                                      sigma=random.choice([0.2, 0.5, 1.0]))
                if random.random() < 0.2 and len(elite) > 1:
                    child = child.migrate(random.choice(elite))
                new_pop.append(child)
            pop = new_pop

    return best_net, best_fit, time.time() - t0


# ============================================================
def main():
    print("=" * 68)
    print("  ModNet v004 — نورون خالص + هرس مغزی")
    print("=" * 68)

    N = 32
    xs = np.linspace(0, 1, N, dtype=np.float32)
    inputs = np.array([encode(float(x)) for x in xs], dtype=np.float32)
    targets = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)

    print(f"  داده: {N} نقطه   ورودی: {N_IN} بیت   خروجی: {N_OUT} بیت")
    print(f"  پیمانه‌ها: {MODULE_SIZES}   کل: {sum(MODULE_SIZES)}")
    print()

    net, fit, dt = evolve(inputs, targets)
    c, r, W_total, W_nz = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0

    print(f"\n  نتیجه: fit={fit:+.5f}   RMSE={rmse:.4f}   زمان={dt:.1f}s")
    print(f"  پیمانه‌ها: {net.mods}   کل: {net.total}")
    print(f"  cross={c}   rec={r}   wires={W_nz}/{W_total} "
          f"({100*W_nz/W_total:.1f}% چگالی)")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for i in range(N):
        xb = inputs[i:i+1]
        out = net.forward_batch(xb, N_STEPS)[0]
        yp = float(out @ POWERS) / DENOM
        print(f"  {xs[i]:6.3f}  {targets[i]:7.3f}  {yp:10.3f}  {abs(yp-targets[i]):7.3f}")


if __name__ == '__main__':
    main()
