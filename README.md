# 🎛️ Wordle Solver using Entropy| Information Theory
<br><br>

## 🚀 Overview

> **"Wordle is a process of reducing uncertainty!"**<br>
> This Python solver plays Wordle by choosing the most informative word at each step to efficiently narrow down possible answers.<br><br>
> **"Core Concept: Entropy as Information Measure"**<br>
> Shannon Entropy: E = - Σ [p(x) * log₂(p(x))]

<br><br>

## 📊 Features
- ✅**Two Solving Stratigies:**
> Uniform Prior: Assumes all remaining words are equally likely<br>
> Frequency-Based Prior: Uses real-world word frequency data for more human-like guessing<br>
- ✅**Performance Optimizations:**
> Pattern precomputation and caching<br>
> Vectorized NumPy operations<br>
> Integer-based pattern representation<br>
  
---
<br><br>
## 🤝🤝🤝 Try it yourself! 🤝🤝🤝
1. Run setup.py
2. Run solver-optimzed-frequency.py
   (Initialization may take a few minutes, since involving matrix precomputation.)
   
---
<br><br>
## 📸 How it looks?

### 1. Performance Test

<div align="center">
  <img src="assets/sigmoid-performance-test.gif" width="600" />
  
  *Testing performance over all Wordle answers.*
</div>

### 2. Interactive Mode

<div align="center">
  <img src="assets/solver-optimized-frequency.gif" width="600" />
  
  *Play Wordle with recommended guesses.*
</div>
