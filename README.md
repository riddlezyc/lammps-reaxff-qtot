# LAMMPS ReaxFF QEq with Total Charge Control (`qtot`)

[![LAMMPS](https://img.shields.io/badge/LAMMPS-stable_29Aug2024-blue)](https://www.lammps.org/)
[![License](https://img.shields.io/badge/license-GPL-green)](LICENSE)

Add a `qtot` keyword to `fix qeq/reaxff` in LAMMPS, allowing users to specify a **non-zero total system charge** for the ReaxFF charge equilibration (QEq/EEM) solver.

## Keywords

`LAMMPS`, `ReaxFF`, `QEq`, `EEM`, `charge equilibration`, `electronegativity equalization`, `qtot`, `total charge`, `net charge`, `fix qeq/reaxff`, `fix qeq/reax`, `dual CG`, `conjugate gradient`, `KOKKOS`, `GPU`, `reactive molecular dynamics`, `ReaxFF charge`, `non-neutral`, `charged system`, `redox`, `electrochemistry`, `battery`, `force field`, `molecular dynamics`, `MD`

## Problem

The standard `fix qeq/reaxff` in LAMMPS forces the system to be charge-neutral (`∑q = 0`). There is no built-in way to specify a non-zero net charge — for example, simulating a charged molecule, a redox-active species, or an electrode with excess charge.

## Solution

A new optional `qtot` keyword is added to `fix qeq/reaxff`:

```lammps
# Neutral system (default, backward-compatible)
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff

# System with total charge +2e
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff qtot 2.0

# Combined with maxiter
fix 1 all qeq/reax 1 0.0 10.0 1e-6 reaxff qtot -1.5 maxiter 500
```

## How It Works

The original QEq solver in LAMMPS uses a **dual conjugate gradient** method for electronegativity equalization (EEM). The total charge constraint is enforced via a Lagrange multiplier:

```
q[i] = s[i] - u * t[i]
u    = (∑s - Q_tot) / ∑t
```

Where `s` and `t` are two CG solutions sharing the same Coulomb matrix. The `qtot` parameter simply replaces `Q_tot = 0` with the user-specified value.

This is **mathematically equivalent** to the standard extended-matrix EEM formulation, but more efficient for parallel computation (no global constraint row in each CG iteration).

## Files Modified

| File | Description |
|---|---|
| `fix_qeq_reaxff.h` | Added `double qtot` member variable |
| `fix_qeq_reaxff.cpp` | Parse `qtot` keyword, modified `calculate_Q()` formula |
| `fix_qeq_reaxff_kokkos.cpp` | Same formula fix for GPU/KOKKOS backend |
| `EEM_QEQ_Derivation.md` | Full mathematical derivation and formulas |

## Installation

Copy the modified files into your LAMMPS source tree:

```bash
cp fix_qeq_reaxff.h fix_qeq_reaxff.cpp \
   /path/to/lammps/src/REAXFF/

cp fix_qeq_reaxff_kokkos.cpp \
   /path/to/lammps/src/KOKKOS/
```

Then rebuild LAMMPS:


## Related Resources

- [LAMMPS Official Documentation](https://docs.lammps.org/fix_qeq_reaxff.html)
