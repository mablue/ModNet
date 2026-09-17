"""
ModNet: شبکه بدون لایه، بازگشتی، پیمانه‌ای
==========================================
- ۴ پیمانه: ۳ اصلی + ۱ مشترک
- هر عضو می‌تونه به هر عضو دیگه وصل بشه (حتی خودش)
- شبکه بازگشتیه: چند گام می‌چرخه، بعد خروجی می‌ده
- پیمانه مشترک: فقط از پیمانه‌های دیگه ورودی می‌گیره
- تکامل: جهش + انتخاب
"""
import random
import time

N_IN = 4          # ورودی: x در [0,1] با 4 بیت
N_OUT = 4         # خروجی: y با 4 بیت
N_STEPS = 6       # چند گام بچرخه
MODULE_SIZES = [4, 4, 4, 4]   # 3 اصلی + 1 مشترک
OUT_MODULE = 3    # خروجی از پیمانه 3 (مشترک)


# ============================================================
#  کدگذاری
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
#  عضو (نورون)
# ============================================================
class Node:
    __slots__ = ('module', 'conns', 'thr')
    def __init__(self, module, conns, thr):
        self.module = module
        self.conns = conns          # لیست (src, weight)
        self.thr = thr              # آستانه

    def clone(self):
        return Node(self.module, list(self.conns), self.thr)


# ============================================================
#  شبکه
# ============================================================
class ModNet:
    def __init__(self, n_in, module_sizes, out_module):
        self.n_in = n_in
        self.mods = list(module_sizes)
        self.out_mod = out_module
        self.total = sum(module_sizes)
        self.nodes = []
        self._init_random()

    def _mod_of(self, i):
        s = 0
        for m, sz in enumerate(self.mods):
            s += sz
            if i < s:
                return m
        return len(self.mods) - 1

    def _init_random(self):
        for i in range(self.total):
            mod = self._mod_of(i)
            n_conn = random.randint(1, 3)
            conns = []
            for _ in range(n_conn):
                r = random.random()
                if mod == self.out_mod:
                    # پیمانه مشترک: فقط از نورون‌ها
                    src = random.randrange(self.total)
                    w = random.choice([-2, -1, 1, 2])
                elif r < 0.3:
                    # ورودی مستقیم
                    src = -random.randint(1, self.n_in)
                    w = random.choice([-1, 1])
                else:
                    # نورون دیگه (هر پیمانه‌ای)
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
        out = []
        for i in range(self.total):
            if self._mod_of(i) == self.out_mod:
                out.append(state[i])
                if len(out) == N_OUT:
                    break
        while len(out) < N_OUT:
            out.append(0)
        return out

    def clone(self):
        n = ModNet.__new__(ModNet)
        n.n_in = self.n_in
        n.mods = list(self.mods)
        n.out_mod = self.out_mod
        n.total = self.total
        n.nodes = [x.clone() for x in self.nodes]
        return n

    def mutate(self, rate=2):
        new = self.clone()
        for _ in range(rate):
            i = random.randrange(new.total)
            nd = new.nodes[i]
            r = random.random()

            if r < 0.35 and nd.conns:
                j = random.randrange(len(nd.conns))
                src, w = nd.conns[j]
                w = max(-3, min(3, w + random.choice([-1, 1])))
                nd.conns[j] = (src, w)

            elif r < 0.65 and nd.conns:
                j = random.randrange(len(nd.conns))
                if nd.module == new.out_mod or random.random() > 0.3:
                    src = random.randrange(new.total)
                else:
                    src = -random.randint(1, new.n_in)
                nd.conns[j] = (src, random.choice([-2, -1, 1, 2]))

            elif r < 0.85:
                nd.thr = max(0, min(3, nd.thr + random.choice([-1, 1])))

            else:
                if len(nd.conns) < 5:
                    if nd.module == new.out_mod or random.random() > 0.3:
                        src = random.randrange(new.total)
                    else:
                        src = -random.randint(1, new.n_in)
                    nd.conns.append((src, random.choice([-2, -1, 1, 2])))
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
#  fitness
# ============================================================
def fitness(net, data):
    err = 0.0
    for x, y in data:
        out = net.forward(encode(x))
        yp = decode(out)
        err += (yp - y) ** 2
    return -err / len(data)


# ============================================================
#  تکامل
# ============================================================
def evolve(data, pop_size=60, gens=400, verbose=True):
    pop = [ModNet(N_IN, MODULE_SIZES, OUT_MODULE) for _ in range(pop_size)]
    best_fit = -1e9
    best_net = None
    t0 = time.time()

    for gen in range(gens):
        scored = sorted([(fitness(n, data), n) for n in pop], key=lambda x: -x[0])
        if scored[0][0] > best_fit + 1e-6:
            best_fit = scored[0][0]
            best_net = scored[0][1].clone()
            if verbose:
                cross, rec = best_net.stats()
                print(f"  gen {gen:4d}   fit={best_fit:+.4f}   cross={cross:2d}   rec={rec:2d}")

        if best_fit > -0.0005:
            break

        elite = [n for _, n in scored[:pop_size // 4]]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            new_pop.append(random.choice(elite).mutate(random.randint(1, 4)))
        pop = new_pop

    return best_net, best_fit, time.time() - t0


# ============================================================
#  main
# ============================================================
def main():
    print("=" * 62)
    print("  ModNet — بدون لایه، بازگشتی، پیمانه‌ای")
    print("  یادگیری y = x²   روی x در [0,1]")
    print("=" * 62)
    print(f"  پیمانه‌ها: {MODULE_SIZES}  (3 اصلی + 1 مشترک)")
    print(f"  کل اعضا: {sum(MODULE_SIZES)}")
    print(f"  هر forward: {N_STEPS} گام بازگشتی")
    print(f"  ورودی: {N_IN} بیت   خروجی: {N_OUT} بیت از پیمانه مشترک")
    print()

    data = [(i / 15, (i / 15) ** 2) for i in range(16)]

    print("  در حال تکامل...")
    net, fit, dt = evolve(data, pop_size=60, gens=400)

    cross, rec = net.stats()
    print(f"\n  نتیجه: fit={fit:+.4f}   زمان={dt:.1f}s")
    print(f"  یال‌های بین‌پیمانه‌ای: {cross}")
    print(f"  یال‌های بازگشتی (حلقه): {rec}")

    print(f"\n  {'x':>6s}  {'هدف':>7s}  {'پیش‌بینی':>9s}")
    for x, y in data:
        yp = decode(net.forward(encode(x)))
        print(f"  {x:6.3f}  {y:7.3f}  {yp:9.3f}")


if __name__ == '__main__':
    main()