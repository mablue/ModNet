"""
ModNet v002 — نورون خالص
========================
- فقط نورون: جمع وزنی + آستانه
- وزن‌ها و bias عدد حقیقی (float)
- پیمانه‌ای، بازگشتی، بدون لایه
- تکامل با جهش گاوسی
"""
import random, math, time

N_IN = 4
N_OUT = 8
N_STEPS = 8
MODULE_SIZES = [6, 6, 6, 6]
OUT_MODULE = 3


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
    __slots__ = ('module', 'conns', 'bias')
    # conns: list of (src:int, w:float)
    # خروجی = 1 if sum(w·x) + bias >= 0 else 0
    def __init__(self, module, conns, bias):
        self.module = module
        self.conns = conns
        self.bias = bias

    def clone(self):
        return Node(self.module, list(self.conns), self.bias)


# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module, out_size=N_OUT):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.total = sum(module_sizes)
        self.out_size = out_size
        self.nodes = []
        self._init()

    def _mod_of(self, i):
        s = 0
        for m, sz in enumerate(self.mods):
            s += sz
            if i < s: return m
        return len(self.mods) - 1

    def _mod_indices(self, m):
        out, s = [], 0
        for mm, sz in enumerate(self.mods):
            if mm == m: return list(range(s, s + sz))
            s += sz
        return out

    def _init(self):
        for i in range(self.total):
            mod = self._mod_of(i)
            n_conn = random.randint(1, 3)
            conns = []
            for _ in range(n_conn):
                if mod == self.out_mod:
                    src = random.randrange(self.total)
                else:
                    src = (random.randrange(self.total) if random.random() > 0.3
                           else -random.randint(1, self.n_in))
                conns.append((src, random.gauss(0, 1)))
            self.nodes.append(Node(mod, conns, random.gauss(0, 1)))

    def forward(self, inputs, steps=N_STEPS):
        state = [0] * self.total
        for _ in range(steps):
            new = list(state)
            for i, nd in enumerate(self.nodes):
                s = nd.bias
                for src, w in nd.conns:
                    if src < 0:
                        s += w * inputs[-src - 1]
                    else:
                        s += w * state[src]
                new[i] = 1 if s >= 0 else 0
            state = new
        idx = self._mod_indices(self.out_mod)
        out = [state[i] for i in idx][:self.out_size]
        while len(out) < self.out_size: out.append(0)
        return out

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in; n.mods = list(self.mods)
        n.out_mod = self.out_mod; n.total = self.total
        n.out_size = self.out_size
        n.nodes = [x.clone() for x in self.nodes]
        return n

    def mutate(self, rate=2, sigma=0.5):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            nd = new.nodes[i]
            r = random.random()

            if r < 0.35 and nd.conns:
                # جهش وزن
                j = random.randrange(len(nd.conns))
                src, w = nd.conns[j]
                nd.conns[j] = (src, w + random.gauss(0, sigma))

            elif r < 0.55:
                # جهش bias
                nd.bias += random.gauss(0, sigma)

            elif r < 0.75 and nd.conns:
                # تغییر سورس
                j = random.randrange(len(nd.conns))
                _, w = nd.conns[j]
                if nd.module == new.out_mod:
                    src = random.randrange(new.total)
                else:
                    src = (random.randrange(new.total) if random.random() > 0.3
                           else -random.randint(1, new.n_in))
                nd.conns[j] = (src, w)

            elif r < 0.90:
                # اضافه کردن اتصال
                if len(nd.conns) < 6:
                    if nd.module == new.out_mod or random.random() > 0.3:
                        src = random.randrange(new.total)
                    else:
                        src = -random.randint(1, new.n_in)
                    nd.conns.append((src, random.gauss(0, 1)))

            else:
                # حذف اتصال
                if len(nd.conns) > 1:
                    nd.conns.pop(random.randrange(len(nd.conns)))
        return new

    def migrate(self, other):
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
                    if self._mod_of(src) != nd.module: cross += 1
                    if src == i: rec += 1
        return cross, rec


# ============================================================
def fitness(net, data):
    e = 0.0
    for inp, tgt in data:
        yp = decode(net.forward(inp))
        e += (yp - tgt) ** 2
    return -e / len(data)


# ============================================================
def evolve(data, pop_size=200, gens=800, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE) for _ in range(pop_size)]
    best_fit, best_net = -1e9, None
    t0 = time.time()
    plateau = 0

    for gen in range(gens):
        scored = sorted([(fitness(n, data), n) for n in pop],
                        key=lambda x: -x[0])
        if scored[0][0] > best_fit + 1e-5:
            best_fit = scored[0][0]
            best_net = scored[0][1].clone()
            plateau = 0
            if verbose:
                c, r = best_net.stats()
                print(f"  gen {gen:4d}   fit={best_fit:+.4f}   cross={c:2d}   rec={r:2d}")
        else:
            plateau += 1

        if plateau > 100:
            if verbose:
                print(f"  gen {gen:4d}   ⚡ restart")
            keep = pop_size // 5
            elite = [n for _, n in scored[:keep]]
            pop = list(elite) + [ModNet(N_IN, MODULE_SIZES, OUT_MODULE)
                                 for _ in range(pop_size - keep)]
            plateau = 0
            continue

        if best_fit > -0.0005: break

        elite = [n for _, n in scored[:pop_size // 5]]
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
    print("  ModNet v002 — نورون خالص (وزن حقیقی + bias حقیقی)")
    print("=" * 64)
    print(f"  پیمانه‌ها: {MODULE_SIZES}   کل: {sum(MODULE_SIZES)}")
    print(f"  خروجی: {N_OUT} بیت   گام: {N_STEPS}   pop=200")
    print()

    data = [(encode(i / 31), math.sin(i / 31 * math.pi * 2) * 0.5 + 0.5)
            for i in range(32)]

    print("  مسئله: y = sin(2πt) نرمال‌شده")
    print("  در حال تکامل...\n")

    net, fit, dt = evolve(data)
    c, r = net.stats()
    rmse = math.sqrt(-fit) if fit < 0 else 0

    print(f"\n  نتیجه: fit={fit:+.5f}   RMSE={rmse:.4f}   زمان={dt:.1f}s")
    print(f"  cross={c}   rec={r}")

    print(f"\n  {'t':>6s}  {'هدف':>7s}  {'پیش‌بینی':>10s}  {'خطا':>7s}")
    for inp, tgt in data:
        yp = decode(net.forward(inp))
        print(f"  {inp:6.3f}  {tgt:7.3f}  {yp:10.3f}  {abs(yp-tgt):7.3f}")


if __name__ == '__main__':
    main()
