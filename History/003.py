"""
ModNet v003 — numpy vectorization + multiprocessing
===================================================
- forward روی همه داده یک‌جا (batch)
- وزن‌ها ماتریسی (N×N)
- ارزیابی جمعیت با همه هسته‌ها
- جهش با numpy
"""
import numpy as np
import random, math, time, os
from multiprocessing import Pool, cpu_count

N_IN = 4
N_OUT = 8
N_STEPS = 8
MODULE_SIZES = [6, 6, 6, 8]
OUT_MODULE = 3
POWERS = np.array([1,2,4,8,16,32,64,128], dtype=np.float32)
DENOM = 255.0


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
        self.W = np.zeros((self.total, self.total), dtype=np.float32)
        self.Win = np.zeros((self.total, self.n_in), dtype=np.float32)
        self.bias = np.zeros(self.total, dtype=np.float32)
        self._mod_map = self._build_mod_map()
        self.out_idx = self._mod_indices(out_module)[:out_size]
        self._init()

    def _build_mod_map(self):
        m = np.zeros(self.total, dtype=np.int32)
        s = 0
        for mod, sz in enumerate(self.mods):
            for i in range(s, s+sz):
                m[i] = mod
            s += sz
        return m

    def _mod_of(self, i):
        return int(self._mod_map[i])

    def _mod_indices(self, m):
        return list(np.nonzero(self._mod_map == m)[0])

    def _init(self):
        # هر نورون ۱ تا ۳ اتصال
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
        """inputs_batch: (B, n_in) numpy. return (B, out_size)."""
        B = inputs_batch.shape[0]
        state = np.zeros((B, self.total), dtype=np.float32)
        const = inputs_batch @ self.Win.T + self.bias   # (B, total)
        Wt = self.W.T
        for _ in range(steps):
            state = ((state @ Wt + const) >= 0).astype(np.float32)
        return state[:, self.out_idx]

    def fitness(self, data_in, data_tgt, steps=N_STEPS):
        out = self.forward_batch(data_in, steps)
        preds = (out @ POWERS) / DENOM
        return float(-np.mean((preds - data_tgt) ** 2))

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = list(self.mods)
        n.out_mod = self.out_mod
        n.total = self.total
        n.out_size = self.out_size
        n._mod_map = self._mod_map
        n.out_idx = self.out_idx
        n.W = self.W.copy()
        n.Win = self.Win.copy()
        n.bias = self.bias.copy()
        return n

    def mutate(self, rate=2, sigma=0.5):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            r = random.random()

            if r < 0.35:
                # جهش وزن داخلی
                j = random.randrange(new.total)
                if new.W[i, j] != 0:
                    new.W[i, j] += random.gauss(0, sigma)
                else:
                    new.W[i, j] = random.gauss(0, 1)

            elif r < 0.55:
                # جهش bias
                new.bias[i] += random.gauss(0, sigma)

            elif r < 0.75:
                # جهش وزن ورودی
                j = random.randrange(new.n_in)
                if new.Win[i, j] != 0:
                    new.Win[i, j] += random.gauss(0, sigma)
                else:
                    new.Win[i, j] = random.gauss(0, 1)

            elif r < 0.90:
                # حذف یه اتصال
                nz = np.nonzero(new.W[i])[0]
                if len(nz) > 1:
                    j = int(random.choice(nz))
                    new.W[i, j] = 0

            else:
                # افزودن اتصال
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

    def stats(self):
        # فقط در زمان گزارش
        Wnz = np.nonzero(self.W)
        cross = 0
        rec = 0
        for i, j in zip(Wnz[0], Wnz[1]):
            if self._mod_map[i] != self._mod_map[j]:
                cross += 1
            if i == j:
                rec += 1
        return cross, rec


# ============================================================
#  worker برای multiprocessing
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
def evolve(data_in, data_tgt, pop_size=200, gens=800, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE, N_OUT)
           for _ in range(pop_size)]

    n_proc = max(1, cpu_count())
    if verbose:
        print(f"  هسته‌ها: {n_proc}   جمعیت: {pop_size}   نسل: {gens}")
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

            if top_fit > best_fit + 1e-5:
                best_fit = top_fit
                best_net = pop[best_idx].clone()
                plateau = 0
                if verbose:
                    c, r = best_net.stats()
                    print(f"  gen {gen:4d}   fit={best_fit:+.5f}   cross={c:3d}   rec={r:3d}")
            else:
                plateau += 1

            if plateau > 100:
                if verbose:
                    print(f"  gen {gen:4d}   ⚡ restart")
                keep = pop_size // 5
                idx = np.argsort(fits)[::-1][:keep]
                elite = [pop[int(i)] for i in idx]
                pop = list(elite) + [ModNet(N_IN, MODULE_SIZES,
                                            OUT_MODULE, N_OUT)
                                     for _ in range(pop_size - keep)]
                plateau = 0
                continue

            if best_fit > -0.0005:
                break

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
    print("=" * 64)
    print("  ModNet v003 — numpy + multiprocessing")
    print("=" * 64)

    N = 32
    xs = np.linspace(0, 1, N, dtype=np.float32)
    inputs = np.array([encode(float(x)) for x in xs], dtype=np.float32)
    targets = (np.sin(xs * 2 * np.pi) * 0.5 + 0.5).astype(np.float32)

    print(f"  داده: {N} نقطه، ورودی {inputs.shape}")
    print(f"  پیمانه‌ها: {MODULE_SIZES}   کل: {sum(MODULE_SIZES)}")
    print()

    net, fit, dt = evolve(inputs, targets)
    c, r = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0

    print(f"\n  نتیجه: fit={fit:+.5f}   RMSE={rmse:.4f}   زمان={dt:.1f}s")
    print(f"  cross={c}   rec={r}")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for i in range(N):
        xb = inputs[i:i+1]
        out = net.forward_batch(xb, N_STEPS)[0]
        yp = float(out @ POWERS) / DENOM
        print(f"  {xs[i]:6.3f}  {targets[i]:7.3f}  {yp:10.3f}  {abs(yp-targets[i]):7.3f}")


if __name__ == '__main__':
    main()
