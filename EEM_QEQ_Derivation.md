# ReaxFF EEM/QEq 双法求解原理与公式推导

## 1. 物理模型：电荷平衡法 (Electronegativity Equalization Method, EEM)

ReaxFF 中原子电荷由电负性均衡原理确定：平衡时体系中每个原子的化学势（能对电荷的偏导）处处相等。

### 1.1 二次型能量展开

体系静电能按电荷做泰勒展开至二阶：

$$
E(q_1,\ldots,q_n) = \sum_{i=1}^{n} \chi_i q_i + \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} J_{ij} \, q_i q_j
$$

| 符号 | 含义 |
|---|---|
| $\chi_i$ | 原子 $i$ 的电负性 (electronegativity) |
| $\eta_i$ | 原子 $i$ 的自硬度 (self-hardness / idempotential) |
| $J_{ij} = 2\eta_i$ (当 $i=j$) | 对角元：自库仑积分，$J_{ii}=2\eta_i=U_i$ |
| $J_{ij}$ (当 $i \neq j$) | 非对角元：屏蔽库仑积分 $J_{ij}=C\cdot T(r)/[r^3+\gamma_{ij}^{-1}]^{1/3}$ |
| $\gamma_{ij}=(\gamma_i\gamma_j)^{-3/2}$ | 屏蔽参数 |

### 1.2 平衡条件

体系受总电荷约束 $\sum_k q_k = Q_{\text{tot}}$，引入拉格朗日乘子 $\lambda$：

$$
\frac{\partial}{\partial q_i}\left[ E(q) - \lambda\left(\sum_k q_k - Q_{\text{tot}}\right) \right] = 0
$$

得到**扩展方程组**（$n+1$ 个方程，$n+1$ 个未知数 $q_1,\ldots,q_n,\lambda$）：

$$
\boxed{
\begin{cases}
\displaystyle\sum_{j=1}^{n} J_{ij}\,q_j + \chi_i + \lambda = 0, & i=1,\ldots,n \\[8pt]
\displaystyle\sum_{k=1}^{n} q_k = Q_{\text{tot}}
\end{cases}}
$$

矩阵形式：

$$
\begin{pmatrix}
J & \mathbf{1} \\ \mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix} \mathbf{q} \\ \lambda \end{pmatrix}
=
\begin{pmatrix} -\boldsymbol{\chi} \\ Q_{\text{tot}} \end{pmatrix}
$$

其中 $J \in \mathbb{R}^{n\times n}$ 对称，$\mathbf{1}=(1,\ldots,1)^\top$。

---

## 2. 方法 A：扩展矩阵直接求解（原始 Python 代码）

### 2.1 矩阵构造

$$
\tilde{J} = \begin{pmatrix}
J_{11} & J_{12} & \cdots & J_{1n} & 1 \\
J_{21} & J_{22} & \cdots & J_{2n} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
J_{n1} & J_{n2} & \cdots & J_{nn} & 1 \\
1 & 1 & \cdots & 1 & 0
\end{pmatrix},
\quad
\mathbf{b} = \begin{pmatrix} -\chi_1 \\ -\chi_2 \\ \vdots \\ -\chi_n \\ Q_{\text{tot}} \end{pmatrix}
$$

其中对角元 $J_{ii} = 2\eta_i$，非对角元 $J_{ij}$ 如下。

### 2.2 屏蔽库仑与 Taper

非对角元 $J_{ij}$（$i\neq j$）：

$$
J_{ij} = \text{Tap}(r_{ij}) \cdot \frac{C}{\big(r_{ij}^3 + \gamma_{ij}^{-1}\big)^{1/3}}
$$

其中 $C = 14.4\;\text{eV·Å·}e^{-2}$，$\gamma_{ij}=(\gamma_i\gamma_j)^{-3/2}$。

7 阶 Taper 多项式（$r\in[0,\text{SWB}]$）：

$$
\text{Tap}(r) = \sum_{k=0}^{7} \text{Tap}[k] \cdot r^k
$$

系数 $\text{Tap}[0..7]$ 由 `init_taper()` 计算，满足 $\text{Tap}(0)=1$，$\text{Tap}(\text{SWB})=0$，且平滑截断。

### 2.3 求解

$$
(\tilde{J})\begin{pmatrix} \mathbf{q} \\ \lambda \end{pmatrix} = \mathbf{b}
\quad\Rightarrow\quad
\begin{pmatrix} \mathbf{q} \\ \lambda \end{pmatrix} = \tilde{J}^{-1}\mathbf{b}
$$

可用 `np.linalg.solve()` 或 CG 直接求解。

### 2.4 优缺点

- 矩阵大小 $n+1$，约束行破坏对称正定性
- 并行化时约束行需要全局通信
- 无历史信息复用

---

## 3. 方法 B：Dual CG 线性叠加（LAMMPS fix qeq/reaxff）

### 3.1 核心思想：线性叠加原理

由于体系是**线性**的，约束方程组的解空间是一个二维仿射空间。通解可以写成两个齐次解的线性组合。

**定义两个基向量 $\mathbf{s}, \mathbf{t}$**：

$$
\begin{aligned}
J\mathbf{s} &= -\boldsymbol{\chi} \quad &\text{(纯电负性驱动解)} \\
J\mathbf{t} &= -\mathbf{1} \quad &\text{(均匀电荷再分配模式)}
\end{aligned}
$$

两方程组共用**同一个矩阵** $J$（$n\times n$，对称正定），仅右端项不同。

### 3.2 线性叠加构造约束解

设最终电荷为：

$$
\mathbf{q} = \mathbf{s} - u \cdot \mathbf{t}
$$

代入约束条件 $\sum q_i = Q_{\text{tot}}$：

$$
\sum_i s_i - u\sum_i t_i = Q_{\text{tot}}
\quad\Rightarrow\quad
u = \frac{\sum_i s_i - Q_{\text{tot}}}{\sum_i t_i}
$$

### 3.3 等价性证明

验证这样构造的 $\mathbf{q}$ 满足原扩展方程组。

将 $\mathbf{q} = \mathbf{s} - u\mathbf{t}$ 代入 $J\mathbf{q} + \boldsymbol{\chi}$：

$$
\begin{aligned}
J\mathbf{q} + \boldsymbol{\chi} &= J(\mathbf{s} - u\mathbf{t}) + \boldsymbol{\chi} \\
&= J\mathbf{s} - u J\mathbf{t} + \boldsymbol{\chi} \\
&= (-\boldsymbol{\chi}) - u(-\mathbf{1}) + \boldsymbol{\chi} \\
&= u \cdot \mathbf{1}
\end{aligned}
$$

即 $J\mathbf{q} + \boldsymbol{\chi} = u \cdot \mathbf{1}$，所有分量相等。

令 $\lambda = -u$，得 $J\mathbf{q} + \boldsymbol{\chi} + \lambda\mathbf{1} = 0$，且 $\sum q_k = Q_{\text{tot}}$（由 $u$ 的构造保证）。

**结论：Dual CG 方法与扩展矩阵方法在数学上严格等价。** 差异仅在于数值实现路径。

### 3.4 算法流程

```
1. 构建 n×n 矩阵 J
      ┌                 ┐
      │ 2η₁  J₁₂  J₁₃ … │
  J = │ J₂₁  2η₂  J₂₃ … │
      │ J₃₁  J₃₂  2η₃ … │
      │  ⋮    ⋮    ⋮   ⋱ │
      └                 ┘

2. 构造两个右端向量
      b_s = [-χ₁,  -χ₂,  …,  -χₙ]^⊤
      b_t = [ -1,    -1,  …,   -1 ]^⊤

3. Jacobi 预条件共轭梯度法求解
      s ← CG(J, b_s, M⁻¹=diag(1/J_ii))
      t ← CG(J, b_t, M⁻¹=diag(1/J_ii))

4. 全局归约求和（仅需一次 MPI_Allreduce）
      S = Σᵢ s[i],    T = Σᵢ t[i]

5. 计算拉格朗日乘子
      u = (S - Q_tot) / T

6. 线性叠加得到最终电荷
      q[i] = s[i] - u · t[i],   i = 1,…,n
```

### 3.5 历史外推加速

LAMMPS 在两个时间步之间保存 $\mathbf{s}$ 和 $\mathbf{t}$ 的历史值（`s_hist`, `t_hist`），下一次 CG 迭代的初始猜测由外推给出：

$$
\begin{aligned}
s_i^{(0)} &= 4\big(s_i^{(\tau-1)} + s_i^{(\tau-3)}\big) - \big(6s_i^{(\tau-2)} + s_i^{(\tau-4)}\big) \quad \text{(三次外推)} \\[4pt]
t_i^{(0)} &= t_i^{(\tau-2)} + 3\big(t_i^{(\tau-1)} - t_i^{(\tau-2)}\big) \quad \text{(二次外推)}
\end{aligned}
$$

---

## 4. 两种方法的对比

| 特性 | 扩展矩阵 (方法 A) | Dual CG (方法 B) |
|---|---|---|
| 矩阵大小 | $(n+1)\times(n+1)$ | $n\times n$ |
| CG 求解次数 | 1 次 | 2 次（**共用同一矩阵**） |
| 约束处理 | 作为矩阵额外行/列 | 事后线性叠加 |
| 矩阵对称性 | 扩展后非正定 | 对称正定 ✓ |
| Jacobi 预条件 | 难以构造 | $M^{-1}=\text{diag}(1/J_{ii})$ |
| 并行通信 | 约束行需全局归约 | `sparse_matvec` 仅需 ghost 交换 |
| 历史加速 | 无 | 外推 $\mathbf{s}$, $\mathbf{t}$ 初始值 |
| MPI Allreduce | 每次 matvec | **仅在 `calculate_Q` 中一次** |
| CPU 实现 | `fix_qeq_reaxff.cpp` (原始) | `fix_qeq_reaxff.cpp` + `fix_qeq_reaxff_kokkos.cpp` (GPU 融合版) |

---

## 5. 添加 `qtot` 参数的代码修改

### 5.1 改动位置

| 文件 | 行号 | 改动内容 |
|---|---|---|
| `fix_qeq_reaxff.h` | 74 | 添加 `double qtot;` 成员变量 |
| `fix_qeq_reaxff.cpp` | 77 | 移除 `narg > 12` 上限 |
| `fix_qeq_reaxff.cpp` | 87 | 初始化 `qtot = 0.0` |
| `fix_qeq_reaxff.cpp` | 103-107 | 新增 `qtot` 关键字解析 |
| `fix_qeq_reaxff.cpp` | 421 | 电荷检查改为 `fabs(qsum - qtot)` |
| `fix_qeq_reaxff.cpp` | 853 | 公式改为 `u = (s_sum - qtot) / t_sum` |
| `fix_qeq_reaxff_kokkos.cpp` | 840 | 公式改为 `delta = (s_sum - qtot) / t_sum` |

### 5.2 数学公式修改

**修改前**（强制电中性 $Q_{\text{tot}}=0$）：

$$
u = \frac{\sum_i s_i}{\sum_i t_i}, \qquad
q_i = s_i - u \cdot t_i
$$

**修改后**（允许指定总电荷 $Q_{\text{tot}}$）：

$$
\boxed{u = \frac{\sum_i s_i - Q_{\text{tot}}}{\sum_i t_i}, \qquad
q_i = s_i - u \cdot t_i}
$$

### 5.3 使用示例

```lammps
# 默认 qtot=0（向后兼容）
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff

# 指定体系总电荷为 +2
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff qtot 2.0

# 结合 maxiter
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff qtot -1.5 maxiter 500
```

### 5.4 Python Dual CG 参考实现

```python
# 设 sysCharge = Q_tot
b_s = np.array([-chi[atom.symbol] for atom in atoms])
b_t = np.full(nAtoms, -1.0)

# Jacobi preconditioner
M = LinearOperator((nAtoms, nAtoms), matvec=lambda x: Hdia_inv * x)

# 两次 CG 求解
s, _ = cg(H, b_s, tol=1e-5, M=M)
t, _ = cg(H, b_t, tol=1e-5, M=M)

# 约束条件
s_sum = np.sum(s)
t_sum = np.sum(t)
u = (s_sum - sysCharge) / t_sum    # ← qtot 在此处引入
q = s - u * t
```

---

## 6. 数值验证（H₂O 分子）

| 原子 | 扩展矩阵 (方法 A) | Dual CG (方法 B) | 差异 |
|---|---|---|---|
| O | -0.59708318 | -0.59708318 | < 1e-9 |
| H₁ | +0.29856772 | +0.29856772 | < 1e-9 |
| H₂ | +0.29851546 | +0.29851546 | < 1e-9 |
| Σq | 0.00000000 | 0.00000000 | — |

参数：$\chi_O=8.5000,\;\chi_H=3.5442,\;\eta_O=8.4783,\;\eta_H=9.3848,\;\gamma_O=1.1000,\;\gamma_H=0.7390$

---

## 7. 关键公式速查

### 非对角元 $J_{ij}$ 的 `calculate_H(r, shld)`

```c
Taper(r) = ((Tap[7]*r + Tap[6])*r + Tap[5])*r + Tap[4])*r + Tap[3])*r + Tap[2])*r + Tap[1])*r + Tap[0]
denom    = (r³ + shld)^(1/3)
J_ij     = Taper(r) · 14.4 / denom
```

其中 `shld = (γ_i · γ_j)^{-1.5}`。

### CG 中的矩阵向量积 `sparse_matvec`

```
o[i] = η[type[i]] · x[i]                           // 对角项
for each neighbor j of i:
    o[i] += J_ij · x[j]                            // 非对角贡献
    o[j] += J_ij · x[i]                            // 对称贡献
```

注意：C++ 中 `sparse_matvec` 的对角贡献为 `η · x`，而扩展矩阵方法的对角元为 $2η$。两者等价，因 C++ 的 `η` 直接取自力场参数 `reax_param.sbp[].eta`，该参数在 ReaxFF 中本就定义为硬度 $U_i=J_{ii}$，无需额外乘 2。
