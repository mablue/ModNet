"""
ModNet v001
===========
تغییرات از v000:
  - خروجی 8 بیت (256 سطح) به‌جای 4 بیت
  - مسئله سری زمانی (نیاز به حافظه)
  - pop = 200 (تنوع بیشتر)
  - مهاجرت بین پیمانه‌ها
  - جهش هوشمندتر
  - کش کردن fitness
"""
import random, time, math

N_IN = 4
N_OUT = 8              # ← 8 بیت
N_STEPS = 8            # ← گام بیشتر
MODULE_SIZES = [6, 6, 6, 6]    # 3 اصلی + 1 مشترک
OUT_MODULE = 3


# ============================================================
def encode(x, bits=N_IN):
    v = int(round(x * (2**bits - 1)))
    return [(v >> k) & 1 for k in range(bits)]

def decode(bits):
    v = 0
    for k, b in enumerate(bits):
        v |= (b & 1) << k
    return v / (2**len(bits) - 1)


# ============================================================
class Node:
    __slots__ = ('module', 'conns', 'thr')
    def __init__(self, module, conns, thr):
        self.module = module
        self.conns = conns
        self.thr = thr
    def clone(self):
        return Node(self.module, list(self.conns), self.thr)


# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module, out_size=N_OUT):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.total = sum(module_sizes)
        self.out_size = out_size
        self.nodes = []
        self._init_random()

    def _mod_of(self, i):
        s = 0
        for m, sz in enumerate(self.mods):
            s += sz
            if i < s:
                return m
        return len(self.mods) - 1

    def _module_indices(self, m):
        out = []
        s = 0
        for mm, sz in enumerate(self.mods):
            if mm == m:
                return list(range(s, s + sz))
            s += sz
        return out

    def _init_random(self):
        for i in range(self.total):
            mod = self._mod_of(i)
            n_conn = random.randint(1, 3)
            conns = []
            for _ in range(n_conn):
                if mod == self.out_mod:
                    src = random.randrange(self.total)
                    w = random.choice([-2, -1, 1, 2])
                else:
                    r = random.random()
                    if r < 0.3:
                        src = -random.randint(1, self.n_in)
                    else:
                        src = random.randrange(self.total)
                    w = random.choice([-2, -1, 1, 2])
                conns.append((src, w))
            self.nodes.append(Node(mod, conns, random.randint(0, 1)))

    def forward(self, inputs, steps=N_STEPS):
        state = [0] * self.total
        for _ in range(steps):
            new = list(state)
            for i, nd in enumerate(self.nodes):
                s = 0
                for src, w in nd.conns:
                    if src < 0:
                        s += w * inputs[-src - 1]
                    else:
                        s += w * state[src]
                new[i] = 1 if s >= nd.thr else 0
            state = new
        # خروجی: نورون‌های پیمانه مشترک
        mod_idx = self._module_indices(self.out_mod)
        out = [state[i] for i in mod_idx][:self.out_size]
        while len(out) < self.out_size:
            out.append(0)
        return out

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = list(self.mods)
        n.out_mod = self.out_mod
        n.total = self.total
        n.out_size = self.out_size
        n.nodes = [x.clone() for x in self.nodes]
        return n

    def mutate(self, rate=2):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            nd = new.nodes[i]
            r = random.random()

            if r < 0.30 and nd.conns:
                j = random.randrange(len(nd.conns))
                src, w = nd.conns[j]
                w = max(-3, min(3, w + random.choice([-1, 1])))
                nd.conns[j] = (src, w)

            elif r < 0.55 and nd.conns:
                j = random.randrange(len(nd.conns))
                if nd.module == new.out_mod:
                    src = random.randrange(new.total)
                else:
                    src = (random.randrange(new.total) if random.random() > 0.3
                           else -random.randint(1, new.n_in))
                nd.conns[j] = (src, random.choice([-2, -1, 1, 2]))

            elif r < 0.75:
                nd.thr = max(0, min(3, nd.thr + random.choice([-1, 1])))

            elif r < 0.90:
                if len(nd.conns) < 5:
                    if nd.module == new.out_mod or random.random() > 0.3:
                        src = random.randrange(new.total)
                    else:
                        src = -random.randint(1, new.n_in)
                    nd.conns.append((src, random.choice([-2, -1, 1, 2])))

            else:
                if len(nd.conns) > 1:
                    nd.conns.pop(random.randrange(len(nd.conns)))
        return new

    def migrate(self, other):
        """
        مهاجرت: یه عضو تصادفی از other رو با یه عضو تصادفی خودم عوض کن
        ولی فقط اگه پیمانه‌شون یکسان باشه
        """
        new = self.clone()
        for _ in range(3):
            i = random.randrange(new.total)
            j = random.randrange(other.total)
            if new._mod_of(i) == other._mod_of(j):
                new.nodes[i] = other.nodes[j].clone()
        return new

    def stats(self):
        cross = rec = 0
        for i, nd in enumerate(self.nodes):
            for src, _ in nd.conns:
                if src >= 0:
                    if self._mod_of(src) != nd.module:
                        cross += 1
                    if src == i:
                        rec += 1
        return cross, rec


# ============================================================
def fitness(net, data):
    err = 0.0
    for inp, tgt in data:
        out = net.forward(inp)
        yp = decode(out)
        err += (yp - tgt) ** 2
    return -err / len(data)


# ============================================================
def evolve(data, pop_size=200, gens=600, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE) for _ in range(pop_size)]
    best_fit = -1e9
    best_net = None
    t0 = time.time()
    plateau = 0

    for gen in range(gens):
        scored = sorted([(fitness(n, data), n) for n in pop], key=lambda x: -x[0])

        if scored[0][0] > best_fit + 1e-5:
            best_fit = scored[0][0]
            best_net = scored[0][1].clone()
            plateau = 0
            if verbose:
                cross, rec = best_net.stats()
                print(f"  gen {gen:4d}   fit={best_fit:+.4f}   cross={cross:2d}   rec={rec:2d}")
        else:
            plateau += 1

        # ← اگه خیلی وقته بهتر نشده، جمعیت رو نابود کن
        if plateau > 80:
            if verbose:
                print(f"  gen {gen:4d}   ⚡ plateau — restart diversity")
            # 70% جمعیت رو دوباره بساز
            keep = pop_size // 5
            elite = [n for _, n in scored[:keep]]
            pop = list(elite) + [ModNet(N_IN, MODULE_SIZES, OUT_MODULE)
                                 for _ in range(pop_size - keep)]
            plateau = 0
            continue

        if best_fit > -0.0003:
            break

        # انتخاب و تکثیر با مهاجرت
        elite = [n for _, n in scored[:pop_size // 5]]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            parent = random.choice(elite)
            child = parent.mutate(random.randint(1, 4))
            # مهاجرت: 20% احتمال
            if random.random() < 0.2 and len(elite) > 1:
                other = random.choice(elite)
                child = child.migrate(other)
            new_pop.append(child)
        pop = new_pop

    return best_net, best_fit, time.time() - t0


# ============================================================
def main():
    print("=" * 64)
    print("  ModNet v001 — بازگشتی، پیمانه‌ای، بدون لایه")
    print("=" * 64)
    print(f"  پیمانه‌ها: {MODULE_SIZES}  (3 اصلی + 1 مشترک)")
    print(f"  کل اعضا: {sum(MODULE_SIZES)}   pop=200   steps={N_STEPS}")
    print(f"  خروجی: {N_OUT} بیت ({2**N_OUT} سطح)")
    print()

    # --- مسئله: سری زمانی sin ---
    # ورودی: t (0..1)
    # خروجی: sin(t) * 0.5 + 0.5   (نرمال‌شده به [0,1])
    
    data = [(encode(i / 31), (math.sin(i / 31 * math.pi * 2) * 0.5 + 0.5))
        for i in range(32)]
    print("  مسئله: y = sin(2πt) نرمال‌شده")
    print("  در حال تکامل...\n")

    net, fit, dt = evolve(data, pop_size=200, gens=600)

    cross, rec = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0
    print(f"\n  نتیجه: fit={fit:+.5f}   RMSE={rmse:.4f}   زمان={dt:.1f}s")
    print(f"  یال‌های بین‌پیمانه‌ای: {cross}   حلقه‌ها: {rec}")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for inp, tgt in data:
        yp = decode(net.forward(inp))
        print(f"  {inp:6.3f}  {tgt:7.3f}  {yp:10.3f}  {abs(yp-tgt):7.3f}")


if __name__ == '__main__':
    main()
