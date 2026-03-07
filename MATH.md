# The Mathematics of Ranked Digit Layouts

## 1. Spaces

A **space** is a finite product of cyclic groups:

$$A = \mathbb{Z}_{n_0} \times \mathbb{Z}_{n_1} \times \cdots \times \mathbb{Z}_{n_{k-1}}$$

An element of $A$ is a coordinate tuple $(c_0, c_1, \ldots, c_{k-1})$ where $0 \le c_i < n_i$.
The **size** of the space is $|A| = \prod_i n_i$.

A space carries optional axis labels — human-readable names for the dimensions. Nothing else.
In particular, a space carries no information about memory layout or digit representation.

---

## 2. Digit Representations

To operate on coordinates algebraically, we decompose each axis into **prime digits**.

### Mixed-radix decomposition

Every integer $n \ge 1$ has a unique prime factorisation $n = p_0 \cdot p_1 \cdots p_{r-1}$.
A value $c \in \mathbb{Z}_n$ is then uniquely represented as a tuple of **digits**:

$$c = d_0 + p_0 \, d_1 + p_0 p_1 \, d_2 + \cdots \qquad d_j \in \mathbb{Z}_{p_j}$$

This is the standard little-endian mixed-radix encoding. The inverse recovers $c$ from
$(d_0, \ldots, d_{r-1})$ via `digits_to_int`.

### The digit stream of a space

Given a space $A = \mathbb{Z}_{n_0} \times \cdots \times \mathbb{Z}_{n_{k-1}}$, each axis
$i$ contributes $r_i = \Omega(n_i)$ digits (the number of prime factors counted with
multiplicity). Concatenating all per-axis digit vectors in a chosen axis order gives a
**flat digit stream** of length $R = \sum_i r_i$.

The map

$$\text{rank}: A \to \mathbb{Z}_{p_0} \times \mathbb{Z}_{p_1} \times \cdots \times \mathbb{Z}_{p_{R-1}}$$

is a bijection. Its inverse is `unrank`.

> **Key point.** The axis ordering and factorisation are a *representation choice*, not a
> property of the space. Two different representation choices for the same space will yield
> different digit streams and therefore different layouts — even with an identity transform
> between them. In this library, representation choices live privately inside
> `RankedDigitLayout`; the `Space` type is purely a coordinate domain.

---

## 3. Transforms

A **transform** $T$ is a map on flat digit streams:

$$T: \mathbb{Z}_{p_0} \times \cdots \times \mathbb{Z}_{p_{R-1}} \;\longrightarrow\; \mathbb{Z}_{q_0} \times \cdots \times \mathbb{Z}_{q_{S-1}}$$

The library provides six primitive transforms, each with its own structure and invertibility.

### 3.1 Identity

$T(\mathbf{d}) = \mathbf{d}$. Requires $R = S$ and the same radices throughout.

A *Refactor* — splitting an axis into finer digits or merging digits into a coarser axis —
is an Identity transform with different source and destination axis groupings. The digit
stream is unchanged; only how its digits are grouped into coordinates differs.

### 3.2 Permute

A bijection defined by a permutation $\sigma$ of digit positions:

$$T(\mathbf{d})_i = d_{\sigma(i)}$$

Inverse: $T^{-1}(\mathbf{d})_i = d_{\sigma^{-1}(i)}$.

Composing two permutations gives $(\sigma_2 \circ \sigma_1)(i) = \sigma_1(\sigma_2(i))$.

**Axis transposition** is a special case: transposing axes $a$ and $b$ in a space corresponds
to permuting the contiguous blocks of digits belonging to each axis.

### 3.3 Project

Drops digit positions not in a keep-set $K \subseteq \{0, \ldots, R-1\}$:

$$T(\mathbf{d})_i = d_{K_i}$$

Non-injective: the **fiber** over any output has size $\prod_{i \notin K} p_i$.
The fiber size is the `kernel_size` of the transform.

### 3.4 Embed

Places the source digits at specific positions in a larger stream, filling the remaining
positions with a constant vector $\mathbf{f}$:

$$T(\mathbf{d})_j = \begin{cases} d_i & \text{if } j = \text{positions}[i] \\ f_j & \text{otherwise} \end{cases}$$

Embed is the right inverse of Project when positions and keep-set coincide.

### 3.5 BlockAffine

Applies an invertible affine map over $\mathbb{Z}_p$ to a selected subset of digit positions
$P = (p_0, \ldots, p_{m-1})$, all sharing the same prime radix $p$:

$$T(\mathbf{d})_{P_i} = \left(\sum_j M_{ij} \, d_{P_j} + c_i\right) \bmod p$$

with $M \in GL(m, \mathbb{Z}_p)$ and offset $c \in \mathbb{Z}_p^m$. Digits not in $P$
pass through unchanged.

Invertibility requires $\det M \not\equiv 0 \pmod{p}$, which is checked at construction time.

Inverse: $T^{-1}(\mathbf{d})_{P_i} = \left(\sum_j (M^{-1})_{ij}(d_{P_j} - c_j)\right) \bmod p$.

**XOR swizzle** is the case $p = 2$, $c = 0$, $M$ any binary invertible matrix. A common
choice for GPU shared memory is $M = I + E_{ij}$ (XOR one bit position into another),
which scatters a column of elements across all memory banks without changing the storage
footprint.

### 3.6 Compose

Sequential application: $(T_2 \circ T_1)(\mathbf{d}) = T_2(T_1(\mathbf{d}))$.

Requires $T_1$'s output radices to equal $T_2$'s input radices. The composition is
associative; together with Identity it forms a monoid on compatible digit streams.

---

## 4. Ranked Digit Layouts

A **ranked digit layout** is a triple $(A, T, B)$ where:

- $A$ is the source space
- $B$ is the destination space
- $T$ is a transform whose input radices match the digit stream of $A$ and whose output
  radices match the digit stream of $B$

The induced coordinate map is:

$$L: A \to B, \qquad L(c) = \text{unrank}_B\!\left(T\!\left(\text{rank}_A(c)\right)\right)$$

When $B = \mathbb{Z}_N$ is a flat one-dimensional space, $L$ gives a flat memory address.
This is the primary use case: a layout tells a compiler exactly which memory location holds
each logical tensor element.

---

## 5. Composition and Inversion

### Composition

Given $L_1: A \to B$ and $L_2: B \to C$, their sequential composition $L_2 \circ L_1: A \to C$
is defined by composing their transforms:

$$T_{L_2 \circ L_1} = T_{L_2} \circ T_{L_1}$$

This requires the digit stream of $B$ to be the same under both $L_1$'s destination
representation and $L_2$'s source representation.

Composition is the fundamental operation for **hardware lowering**: if $H: \text{hw} \to \text{tile}$
describes which tile element each hardware thread handles, and $L: \text{tile} \to \text{flat}$
is the memory layout of the tile, then $L \circ H: \text{hw} \to \text{flat}$ is the
lowered layout that each thread uses to compute its memory address directly.

### Inversion

A layout $L: A \to B$ is invertible when $T$ is. The inverse $L^{-1}: B \to A$ satisfies
$L^{-1} \circ L = \text{id}_A$.

**Algebraic division.** Given $\text{Logical}: \text{tile} \to \text{flat}$ and $H: \text{hw} \to \text{tile}$, the composed layout $\text{Logical} \circ H: \text{hw} \to \text{flat}$ is obtained directly. Conversely, given the composed layout and $H$, the original logical layout can be recovered as $(\text{Logical} \circ H) \circ H^{-1} = \text{Logical}$.

---

## 6. Analytical Contiguity

For a layout $L: A \to \mathbb{Z}_N$ and a logical axis $a$, the **contiguity** along $a$ is
the largest $w$ such that, from the all-zero base coordinate, stepping through the first $w$
elements along axis $a$ produces consecutive flat addresses.

It is computed digit by digit, from the least-significant digit of axis $a$ upward:

1. Let $\mathbf{0}$ be the all-zero digit vector. Evaluate the base flat address $f_0 = \text{flat}(\mathbf{0})$.
2. For each digit position $d$ belonging to axis $a$, with radix $r_d$:
   - Set digit $d$ to 1 (all others 0) and evaluate $f_1$.
   - If $f_1 - f_0 \neq w_{\text{expected}}$: stop.
   - Verify linearity: $f_{r-1} - f_0 = w_{\text{expected}} \cdot (r_d - 1)$.
   - If linear: multiply contiguity width by $r_d$; update expected stride by $r_d$.
3. Return the accumulated width.

This runs in $O(r_a)$ transform applications, where $r_a = \Omega(n_a)$ is the number of
prime factors of the axis size. It gives the vectorisation width a compiler can exploit
without any runtime scan over the tensor.

**Example.** A right-major layout over $\mathbb{Z}_4 \times \mathbb{Z}_4$ with $N = 16$:
- Axis 1 (column): two binary digits, each with stride 1 then 2 → contiguity = 4.
- Axis 0 (row): stride at the first digit is 4 ≠ 1 → contiguity = 1.

---

## 7. The Digit Permutation for GPU Warp Tiles

A common pattern in GPU kernels: a warp of 32 threads covers a rectangular tile, and each
thread owns a small sub-tile. The mapping from thread state $(lane, t_{row}, t_{col})$ to
tile coordinate $(row, col)$ is given by a formula involving bit-shifts and additions.

Because all axis sizes are powers of 2, every arithmetic operation in such a formula
corresponds to a rearrangement of bits — a **Permute** of the binary digit stream.

Given the thread assignment formula, the permutation order $\sigma$ is derived analytically:
for each output bit position $i$, find which input bit it equals and set $\sigma(i)$ to
that position. The resulting `Transform::Permute` encodes the entire warp layout as a
single bijection on digit streams — no branches, no modular arithmetic at runtime.

The hardware layout $H$ composed with the logical layout $L$ then gives each thread a
closed-form address computation that a compiler can unroll and vectorise using the
analytical contiguity of the composed layout.

---

## 8. Summary of Algebraic Properties

| Transform   | Invertible | Kernel size | Composes with        |
|-------------|-----------|-------------|----------------------|
| Identity    | yes        | 1           | any compatible T     |
| Refactor    | yes        | 1           | any compatible T     |
| Permute     | yes        | 1           | Permute → fused      |
| Project     | no         | > 1         | any (left only)      |
| Embed       | no         | 1           | any (right only)     |
| BlockAffine | yes        | 1           | BlockAffine (same P) |
| Compose     | iff all    | product     | Compose → flattened  |

Two transforms are **extensionally equal** if they agree on every input digit vector.
`simplify()` uses this to collapse inverse pairs, fuse consecutive Permutes, and merge
adjacent Refactors, reducing a composed chain to its canonical form.
