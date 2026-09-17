## ModNet

A Self-Growing Layer-Free Recurrent Architecture for Universal Computation

ModNet is an evolutionary framework that discovers network topology from first principles — no predefined layers, no imposed modular grouping, and no architectural priors. Every node is a general-purpose threshold unit that may receive input from any other node (including itself), and structure emerges spontaneously as the task demands.

This repository contains two complementary implementations of the ModNet idea, each in its own subdirectory with a dedicated README:

---

### 📁 Repository Structure

```
.
├── ModNet/          # Modular variant with emergent cross-module connectivity
│   └── README.md
└── ModNet-Flat/     # Fully layer-free, module-free variant
    └── README.md
```

---

### 🧠 ModNet (Modular)

A Self-Growing Modular Recurrent Architecture for Layer-Free Computation

The original ModNet explores whether cross-module connectivity, recurrence, and sparse wiring emerge spontaneously when the network is allowed to grow, prune, and reorganize itself through a compact evolutionary loop.

· Architecture: no predefined layers; nodes are general-purpose threshold units; any-to-any connectivity including self-loops.
· Tasks: Boolean logic (XOR, 4-bit and 6-bit parity) and continuous regression (sinusoids, quadratic surfaces).
· Results:
  · ✅ 100% accuracy on XOR and 4-bit parity
  · ✅ RMSE < 0.05 on continuous targets
  · ✅ Fewer than 30 nodes per network
· Key finding: structural principles typically imposed by human designers (modularity, recurrence, sparsity) emerge on their own from the task demands — with no explicit regularization or architectural prior.

📄 See ModNet/README.md for full details.

---

### 🧩 ModNet-Flat

A Self-Growing Layer-Free Recurrent Network for Universal Computation

ModNet-Flat strips away even the notion of modules: every node is identical and general-purpose, and the first n_out nodes are designated as outputs. It is evaluated on a much broader benchmark suite to test universality.

· Architecture: fully layer-free and module-free; any-to-any connectivity including self-loops.
· Tasks: 16 benchmarks spanning Boolean logic, digital arithmetic, sorting networks, and continuous regression (full adder, 2-bit adder, comparator, 4-to-1 MUX, 4-bit sorting network, x² + y² surface, multiplication, Gaussian, …).
· Results:
  · ✅ 14 / 16 tasks solved with exact 100% accuracy
  · ✅ No more than 27 nodes for any solved task
  · ✅ RMSE as low as 0.0143 (x² + y² surface) and 0.0254 (multiplication) with only 13 nodes
· Key finding: recurrence emerges spontaneously and scales with task complexity — XOR-2 needs only one self-loop, while Gaussian and multiplication require 10–14 self-loops. Temporal memory appears to be a structural necessity, not an imposed design.

📄 See ModNet-Flat/README.md for full details.

---

### 🔬 Shared Principles

Both variants share the same core idea and the same conclusions:

Principle Description
Layer-free No predefined depth, width, or block structure.
Self-growing Topology is discovered through a compact evolutionary loop (grow / prune / reorganize).
Recurrent by default Any node may connect to any other node, including itself.
Sparse & modular Wiring sparsity and modularity emerge from the task, not from a prior.
Minimal prior No explicit regularization, no architectural bias — only evolution.

"A fully layer-free architecture driven by a minimal evolutionary process can rediscover structural principles — non-linearity, recurrence, sparsity — that are typically imposed by human designers."

---

### 🚀 Getting Started

Each subproject is self-contained and includes its own instructions, source code, and reproducible benchmark suite. Start with the README of whichever variant matches your interest:

· 👉 ModNet/README.md — modular, focused on emergent structure.
· 👉 ModNet-Flat/README.md — fully flat, focused on universal computation across 16 benchmarks.

---

### 📚 Keywords

neuroevolution · self-growing networks · layer-free architectures · modular recurrent networks · recurrent networks · Boolean function learning · digital circuits · sparse connectivity · universal computation