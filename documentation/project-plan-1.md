# Project Plan: An Information-Theoretic Wordle Solver
The goal of this project is to develop a Python program that plays Wordle by strategically maximizing the information gained from each guess. The core idea is to treat the game as a process of reducing uncertainty, using concepts from information theory to choose the most informative word at each step.

## Core Concept: Entropy as a Measure of Information
In Wordle, we start with a large set of possible answers. Each guess we make can result in a different pattern of feedback (comprising grey, yellow, and green tiles). A good guess is one that, on average, splits the remaining possible words into many small, distinct groups. This is where information entropy comes in.

Entropy measures the average "surprise" or information content. We can calculate the expected information, E, for a guess using the formula:

>  E = - Σ [p(x) * log₂(p(x))] 

Where x represents every possible feedback pattern, and p(x) is the probability of that pattern occurring, given the current list of possible answers.

A high-entropy word is one that is unpredictable and will, on average, rule out the largest number of possible answers, regardless of the actual solution.

## Program Versions
I plan to implement two versions to explore the impact of word frequency:

- Version 1 (Uniform Prior): Assumes all remaining possible words are equally likely.

- Version 2 (Frequency-Based Prior): Uses real-world word frequency data (e.g., from a corpus like the British National Corpus) to weight the probability of each word, making the solver more reflective of common usage.

## Program Workflow
The solver will operate in a loop until the answer is found or the guesses are exhausted:

1. Initialization: Load the official list of Wordle answers.

2. Guess Selection:
a. From the current list of possible answers, calculate the entropy for each candidate word.
b. The entropy calculation involves:
  - For a given candidate word, determine the feedback pattern it would produce against every word in the current possible answer set.
  - Group the possible answers by the resulting pattern.
  - Calculate p(x) for each pattern as (size of group) / (total possible words).
  - Compute the entropy using the formula above.
c. Select the word with the highest entropy.

3. Feedback & Filtering:
a. Submit the chosen word and receive the feedback pattern (from the actual game or a simulated answer).
b. Filter the list of possible answers, keeping only those consistent with the new feedback.

4. Repeat: Go back to Step 2 until the answer is found.

## Implementation Subtasks
To build this, the project will be broken down into the following modules:

- Data Acquisition: Fetch and parse the official list of Wordle answers from the game's source code.

- Core Logic:
a. Pattern Function: A function that, given a guess and a target answer, returns the Wordle feedback pattern (e.g., as a tuple of states: ['grey', 'grey', 'green', 'yellow', 'grey']).
b. Entropy Calculator: A function that takes the current list of possible answers and a candidate guess, and returns the calculated entropy.
c. List Filter: A function that, given a guess, the resulting pattern, and the current word list, returns the new, filtered list of possible answers.

- Solver Engine: The main loop that integrates the core logic to play the game step-by-step.

- Visualization & Interface:

- Console Output: At each step, display the top 10 candidate words with their entropies, the current number of remaining possible answers, and the chosen guess.

- Graphical Interface (Stretch Goal): Create a simple GUI that resembles the Wordle game board to visually track the solver's progress.
