# Problem: Entropy calculation is O(n²) complexity.
With 14,855 words means ~220 million pattern calculations, which is too slow. 

# Optimization Approaches:

1. Pattern Precomputation: All ~220 million patterns are computed once and stored as integers in a NumPy matrix

2. Vectorized Operations: Use NumPy for filtering and entropy calculations instead of Python loops

3. Integer Patterns: Convert patterns to base-3 integers for faster comparison

4. Index-based Operations: Work with word indices instead of strings
