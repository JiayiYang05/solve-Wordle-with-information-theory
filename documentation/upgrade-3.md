# Performance bottleneck at 4.5 guesses/word
This is only slightly above human average.
The insight here is that we naturally guess more frequent words, while the algorithm treats all words equally (i.e. assume uniform distribution).
Therefore, we consider frequecy integration to model real-world word likelihood.

# Frequency Integration:
1. For Entropy Calculation:
Use weighted probabilities: p(pattern) = sum(frequencies of words that give this pattern) / total frequency

2. For Guess Selection :
Don't add entropy + frequency directly (different units/scales)

Better approaches:

- Two-stage filtering: Pick top N by entropy, then choose highest frequency

- Weighted score: score = α × entropy + β × log(frequency)

- Frequency as tie-breaker: When entropies are close, pick more common word
