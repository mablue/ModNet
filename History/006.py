"""
ModNet v006 — مکانیزم‌های طبیعی تکامل
======================================
- تناسل جنسی (crossover)
- سالمندی (aging)
- جهش تطبیقی (adaptive mutation)
- تنوع رفتاری (behavioral diversity)
- نخبه‌چرخشی (elite rotation)
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
W_THRESH = 0.05
ACT_LO = 0.02
ACT_HI = 0.98
PRUNE_EVERY = 25

# پارامترهای طبیعی جدید
MAX_AGE = 40              # سن بازنشستگی
CROSSOVER_RATE = 0.5      # احتمال تناسل جنسی
SIGMA_BASE = 0.4          # جهش پایه
SIGMA_GROWTH = 1.3        # ضریب رشد جهش در فلات
SIGMA_MAX = 3.0           # سقف جهش
PLATEAU_TRIGGER = 30      # از چند نسل گیر، جهش زیاد بشه


def encode(x, bits=N_IN):
    v = int(round(x * (2**bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]


# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module, out_size):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.out_size = out_size
        self.age = 0
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

    def behavior(self, data_in, steps=N_STEPS):
        """بردار رفتار: خروجی روی همه داده"""
        out = self.forward_batch(data_in, steps)
        return (out @ POWERS) / DENOM

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = list(self.mods)
        n.out_mod = self.out_mod
        n.out_size = self.out_size
        n.total = self.total
        n.age = self.age
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        n._mod_map = self._mod_map.copy()
        n.out_idx = self.out_idx.copy()
        return n

    def mutate(self, rate=2, sigma=0.5):
        new = self.clone()
        new.age = 0  # فرزند، تازه متولد
        for _ in range(rate):
            i = random.randrange(new.total)
            r = random.random()
            if r < 0.35:
                j = random.randrange(new.total)
                new.W[i, j] += random.gauss(0, sigma)
            elif r < 0.55:
                new.bias[i] += random.gauss(0, sigma)
            elif r < 0.75:
                j = random.randrange(new.n_in)
                new.Win[i, j] += random.gauss(0, sigma)
            elif r < 0.90:
                nz = np.nonzero(new.W[i])[0]
                if len(nz) > 1:
                    new.W[i, int(random.choice(nz))] = 0
            else:
                new.W[i, random.randrange(new.total)] = random.gauss(0, 1)
        return new

    def migrate(self, other):
        new = self.clone()
        new.age = 0
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

    # ========================================================
    #  تناسل جنسی
    # ========================================================
    def crossover(self, other):
        """
        فرزند از دو والد.
        هر نورون: یا از والد اول یا از والد دوم.
        """
        new = self.clone()
        new.age = 0
        if self.total != other.total or self.mods != other.mods:
            # ساختار متفاوت — نمی‌شه ترکیب کرد
            return new
        for i in range(new.total):
            if random.random() < 0.5:
                new.W[i] = other.W[i].copy()
                new.Win[i] = other.Win[i].copy()
                new.bias[i] = other.bias[i]
        return new

    # ========================================================
    #  هرس (همون v005)
    # ========================================================
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

    def try_prune(self, data_in, data_tgt, steps=N_STEPS, min_size=MIN_MODULE_SIZE):
        old_fit = self.fitness(data_in, data_tgt, steps)
        candidate = self.clone()
        removed = candidate._prune_inplace(data_in, min_size)
        if removed == 0:
            return None
        new_fit = candidate.fitness(data_in, data_tgt, steps)
        if new_fit >= old_fit - 1e-6:
            return candidate, removed, new_fit - old_fit
        return None

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
def evolve(data_in, data_tgt, pop_size=200, gens=1500, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT) for _ in range(pop_size)]

    n_proc = max(1, cpu_count())
    if verbose:
        print(f"  هسته‌ها: {n_proc}   جمعیت: {pop_size}   نسل: {gens}")
        print(f"  مکانیزم‌ها: crossover + aging + adaptive mutation")
        print()

    best_fit = -1e9
    best_net = None
    t0 = time.time()
    plateau = 0
    sigma = SIGMA_BASE
    chunk = max(1, pop_size // (n_proc * 4))

    # نسل‌های تولد
    birth_gen = [0] * pop_size

    with Pool(n_proc, initializer=_init_worker,
              initargs=(data_in, data_tgt)) as pool:
        for gen in range(gens):
            fits = pool.map(_fitness_worker, pop, chunksize=chunk)

            # ── سالمندی: هر ۵ نسل، یه جریمه کوچیک برای قدیمی‌ها ──
            for i, n in enumerate(pop):
                n.age = gen - birth_gen[i]
            penalized = [fits[i] - 0.00005 * min(pop[i].age, 200) for i in range(pop_size)]

            best_idx = int(np.argmax(penalized))
            top_fit = fits[best_idx]  # بدون جریمه

            improved = False
            if top_fit > best_fit + 1e-5:
                best_fit = top_fit
                best_net = pop[best_idx].clone()
                plateau = 0
                sigma = SIGMA_BASE
                improved = True
            else:
                plateau += 1
                # ── جهش تطبیقی ──
                if plateau > PLATEAU_TRIGGER:
                    sigma = min(SIGMA_MAX, sigma * SIGMA_GROWTH)
                    plateau = 0

            # ── هرس امن ──
            if gen > 0 and gen % PRUNE_EVERY == 0:
                idx_sort = np.argsort(penalized)[::-1]
                elite_n = min(pop_size // 5, 30)
                total_removed = 0
                total_accepted = 0
                for k in idx_sort[:elite_n]:
                    k = int(k)
                    result = pop[k].try_prune(data_in, data_tgt, N_STEPS, MIN_MODULE_SIZE)
                    if result is not None:
                        candidate, removed, gain = result
                        pop[k] = candidate
                        fits[k] = candidate.fitness(data_in, data_tgt, N_STEPS)
                        penalized[k] = fits[k]
                        total_removed += removed
                        total_accepted += 1
                cur_best_idx = int(np.argmax(penalized))
                if fits[cur_best_idx] > best_fit + 1e-5:
                    best_fit = fits[cur_best_idx]
                    best_net = pop[cur_best_idx].clone()
                    improved = True
                elif (abs(fits[cur_best_idx] - best_fit) <= 1e-5
                      and pop[cur_best_idx].total < best_net.total):
                    best_net = pop[cur_best_idx].clone()

            if verbose and improved:
                c, r, W_tot, W_nz = best_net.stats()
                print(f"  gen {gen:4d}   fit={best_fit:+.5f}   "
                      f"cross={c:3d} rec={r:3d}   size={best_net.total}   "
                      f"σ={sigma:.2f}")

            if best_fit > -0.0005: break

            # ── تولید نسل بعد ──
            idx = np.argsort(penalized)[::-1]
            elite_n = pop_size // 5
            elite = [pop[int(i)] for i in idx[:elite_n]]

            new_pop = []
            new_birth = []
            # نخبه‌ها با سن حفظ می‌شن
            for e in elite:
                new_pop.append(e.clone())
                new_birth.append(birth_gen[pop.index(e)] if e in pop else gen)

            while len(new_pop) < pop_size:
                r = random.random()
                if r < CROSSOVER_RATE and len(elite) >= 2:
                    # تناسل جنسی
                    p1, p2 = random.sample(elite, 2)
                    child = p1.crossover(p2)
                    child = child.mutate(random.randint(1, 3), sigma=sigma)
                elif r < 0.9:
                    # جهش
                    parent = random.choice(elite)
                    child = parent.mutate(random.randint(1, 4), sigma=sigma)
                else:
                    # مهاجرت
                    parent = random.choice(elite)
                    other = random.choice(elite)
                    child = parent.migrate(other)
                new_pop.append(child)
                new_birth.append(gen)

            pop = new_pop
            birth_gen = new_birth

    return best_net, best_fit, time.time() - t0


# ============================================================
def main():
    print("=" * 68)
    print("  ModNet v006 — تناسل + پیری + جهش تطبیقی")
    print("=" * 68)

    N = 32
    xs = np.linspace(0, 1, N, dtype=np.float32)
    inputs = np.array([encode(float(x)) for x in xs], dtype=np.float32)
    targets = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)

    print(f"  داده: {N} نقطه   ورودی: {N_IN} بیت   خروجی: {N_OUT} بیت")
    print(f"  پیمانه‌ها: {MODULE_SIZES}   کل: {sum(MODULE_SIZES)}")
    print()

    net, fit, dt = evolve(inputs, targets)
    c, r, W_tot, W_nz = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0

    print(f"\n  نتیجه: fit={fit:+.5f}   RMSE={rmse:.4f}   زمان={dt:.1f}s")
    print(f"  پیمانه‌ها: {net.mods}   کل: {net.total}")
    print(f"  cross={c}   rec={r}   wires={W_nz}/{W_tot}")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for i in range(N):
        out = net.forward_batch(inputs[i:i+1], N_STEPS)[0]
        yp = float(out @ POWERS) / DENOM
        print(f"  {xs[i]:6.3f}  {targets[i]:7.3f}  {yp:10.3f}  {abs(yp-targets[i]):7.3f}")


if __name__ == '__main__':
    main()
