# Problem: Recompute pattern matrix for each program run
Extremely time consuming, especially during testing stage where code is frequently changed.

# Solution: Cache the matrix
Only pay the computation cost once! The cache files will persist between program runs, making subsequent startups nearly instant.

# Key Features:
- Automatic Cache Detection: Checks if cache exists and is valid

- Word List Change Detection: Recomputes if the word list changes

- Fast Loading: Subsequent runs load instantly from disk

- Cache Management: Easy to clear cache if needed
