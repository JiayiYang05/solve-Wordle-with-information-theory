import requests
import math
from collections import Counter
from typing import List, Tuple, Dict
import random

class WordleSolver:
    def __init__(self, use_frequency=False):
        self.words = self.get_wordle_answers()
        self.use_frequency = use_frequency
        if use_frequency:
            self.word_frequencies = self.load_word_frequencies()
        else:
            self.word_frequencies = {word: 1 for word in self.words}
        
        self.total_frequency = sum(self.word_frequencies.values())
        self.pattern_cache = {}
        
    def get_wordle_answers(self) -> List[str]:
        """Get a smaller subset of Wordle answers for testing."""
        # Using a smaller subset for testing - first 100 words
        url = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"
        response = requests.get(url)
        all_words = [word.strip().upper() for word in response.text.split('\n') if len(word.strip()) == 5]
        return all_words[:100]  # Only use first 100 words for testing
    
    def load_word_frequencies(self) -> Dict[str, float]:
        """Load word frequencies (simplified version)."""
        return {word: 1 for word in self.words}
    
    def get_pattern(self, guess: str, answer: str) -> str:
        """Calculate the Wordle pattern for a guess against an answer."""
        cache_key = (guess, answer)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        pattern = ['X'] * 5
        answer_chars = list(answer)
        guess_chars = list(guess)
        
        # First pass: mark greens
        for i in range(5):
            if guess_chars[i] == answer_chars[i]:
                pattern[i] = 'G'
                answer_chars[i] = None
                guess_chars[i] = None
        
        # Second pass: mark yellows
        for i in range(5):
            if guess_chars[i] is not None and guess_chars[i] in answer_chars:
                pattern[i] = 'Y'
                answer_chars[answer_chars.index(guess_chars[i])] = None
        
        result = ''.join(pattern)
        self.pattern_cache[cache_key] = result
        return result
    
    def filter_words(self, words: List[str], guess: str, pattern: str) -> List[str]:
        """Filter possible words based on guess and pattern."""
        return [word for word in words if self.get_pattern(guess, word) == pattern]
    
    def calculate_entropy_for_guess(self, guess: str, possible_answers: List[str]) -> float:
        """Calculate the entropy of a guess given possible answers."""
        pattern_counts = Counter()
        
        for answer in possible_answers:
            pattern = self.get_pattern(guess, answer)
            pattern_counts[pattern] += 1
        
        if len(possible_answers) == 0:
            return 0
            
        entropy = 0
        for count in pattern_counts.values():
            p = count / len(possible_answers)
            entropy -= p * math.log2(p)
        
        return entropy
    
    def get_best_guesses(self, possible_answers: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """Get the top N guesses with highest entropy."""
        n_answers = len(possible_answers)
        
        # For very small sets, just return the answers
        if n_answers <= 2:
            return [(word, math.log2(n_answers)) for word in possible_answers]
        
        # Calculate entropy only for words that are in the possible answers
        word_scores = []
        for word in possible_answers:
            entropy = self.calculate_entropy_for_guess(word, possible_answers)
            word_scores.append((word, entropy))
        
        # Sort by entropy and return top N
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_n]
    
    def solve_interactive(self):
        """Interactive solver that automatically calculates patterns."""
        # First, select a random answer
        answer = random.choice(self.words)
        possible_answers = self.words.copy()
        guesses_made = []
        
        print("Wordle Solver - Interactive Mode")
        print("I've selected a secret word. Try to guess it!")
        print(f"Total possible answers: {len(possible_answers)}")
        print("Patterns will be shown automatically (G=Green, Y=Yellow, X=Grey)")
        
        for attempt in range(6):
            print(f"\n--- Attempt {attempt + 1} ---")
            print(f"Remaining possible answers: {len(possible_answers)}")
            
            if len(possible_answers) <= 15:
                print(f"Possible answers: {', '.join(possible_answers)}")
            elif len(possible_answers) <= 100:
                print(f"Sample answers: {', '.join(possible_answers[:10])}...")
            
            print("Calculating best guesses...")
            
            # Get best guesses - ONLY from possible answers
            best_guesses = self.get_best_guesses(possible_answers)
            
            print("\nTop recommendations:")
            for i, (word, entropy) in enumerate(best_guesses[:10]):
                print(f"{i+1:2d}. {word} (entropy: {entropy:.3f})")
            
            # Get user input
            while True:
                guess = input("\nEnter your guess (or 'quit' to exit, 'auto' for top recommendation, 'answer' to reveal): ").strip().upper()
                if guess == 'QUIT':
                    return
                elif guess == 'ANSWER':
                    print(f"The answer is: {answer}")
                    continue
                elif guess == 'AUTO':
                    guess = best_guesses[0][0]
                    print(f"Using automatic guess: {guess}")
                    break
                elif len(guess) != 5:
                    print("Guess must be 5 letters!")
                    continue
                elif not guess.isalpha():
                    print("Guess must contain only letters!")
                    continue
                elif guess not in possible_answers:  # Changed from self.words to possible_answers
                    print(f"❌ '{guess}' is not in the remaining possible answers! Try one of the recommendations.")
                    # Show the recommendations again for convenience
                    print("Valid recommendations:", ', '.join([word for word, _ in best_guesses[:5]]))
                    continue
                else:
                    break
            
            # Automatically calculate pattern against the answer
            pattern = self.get_pattern(guess, answer)
            guesses_made.append((guess, pattern))
            
            print(f"Pattern: {pattern}")
            
            # Show the guess with colors for better visualization
            self.display_colored_guess(guess, pattern)
            
            # Filter possible answers
            previous_count = len(possible_answers)
            possible_answers = self.filter_words(possible_answers, guess, pattern)
            
            print(f"Filtered from {previous_count} to {len(possible_answers)} possible answers")
            
            # Check if solved
            if pattern == 'GGGGG':
                print(f"\n🎉 Congratulations! You solved it in {attempt + 1} attempts!")
                print(f"The answer was: {answer}")
                self.display_statistics(guesses_made, answer)
                return
            elif len(possible_answers) == 1:
                print(f"\nOnly one possibility remains: {possible_answers[0]}")
        
        print(f"\nGame over! The answer was: {answer}")
        print("Remaining possible answers:")
        if len(possible_answers) <= 20:
            print(', '.join(possible_answers))
    
    def display_colored_guess(self, guess: str, pattern: str):
        """Display the guess with colored output for better visualization."""
        colors = {
            'G': '🟩',  # Green
            'Y': '🟨',  # Yellow  
            'X': '⬜'   # Grey
        }
        
        colored_output = ''.join(colors[char] for char in pattern)
        print(f"Visual: {colored_output}")
        
        # Also show letter-by-letter coloring
        print("Letters: ", end="")
        for i, (letter, color) in enumerate(zip(guess, pattern)):
            if color == 'G':
                print(f"\033[92m{letter}\033[0m", end=" ")  # Green
            elif color == 'Y':
                print(f"\033[93m{letter}\033[0m", end=" ")  # Yellow
            else:
                print(f"\033[90m{letter}\033[0m", end=" ")  # Grey
        print()
    
    def display_statistics(self, guesses_made, answer):
        """Display statistics about the solution."""
        print(f"\n=== Statistics ===")
        print(f"Solved in {len(guesses_made)} attempts")
        print(f"Answer: {answer}")
        
        print("\nGuess history:")
        for i, (guess, pattern) in enumerate(guesses_made, 1):
            colored = ''.join(['🟩' if c == 'G' else '🟨' if c == 'Y' else '⬜' for c in pattern])
            print(f"Attempt {i}: {guess} -> {pattern} {colored}")
        
        initial_entropy = math.log2(len(self.words))
        print(f"\nInitial entropy: {initial_entropy:.2f} bits")
        print(f"Information gained per guess: {initial_entropy/len(guesses_made):.2f} bits")

    def solve_for_answer(self, answer: str, max_attempts: int = 6) -> List[Tuple[str, str]]:
        """Automatically solve for a specific answer (for testing)."""
        possible_answers = self.words.copy()
        guesses_made = []
        
        print(f"Solving for answer: {answer}")
        
        for attempt in range(max_attempts):
            print(f"Remaining possibilities: {len(possible_answers)}")
            
            # Get best guess - ONLY from possible answers
            best_guesses = self.get_best_guesses(possible_answers, top_n=1)
            guess = best_guesses[0][0]
            
            # Get pattern for this guess
            pattern = self.get_pattern(guess, answer)
            guesses_made.append((guess, pattern))
            
            print(f"Attempt {attempt + 1}: {guess} -> {pattern}")
            self.display_colored_guess(guess, pattern)
            
            # Check if solved
            if pattern == 'GGGGG':
                print(f"🎉 Solved in {attempt + 1} attempts!")
                return guesses_made
            
            # Filter possible answers
            possible_answers = self.filter_words(possible_answers, guess, pattern)
            
            if len(possible_answers) == 0:
                print("No possible answers remain!")
                return guesses_made
        
        print(f"Failed to solve in {max_attempts} attempts")
        return guesses_made

def main():
    print("Wordle Solver using Information Theory")
    print("=" * 40)
    
    # Choose version
    while True:
        version = input("Choose version (1 for uniform distribution, 2 for frequency-based): ").strip()
        if version in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    use_frequency = (version == "2")
    
    # Initialize solver
    print("Initializing Wordle Solver...")
    try:
        solver = WordleSolver(use_frequency=use_frequency)
        print(f"Loaded {len(solver.words)} possible answers")
        print(f"Sample words: {', '.join(solver.words[:5])}...")
    except Exception as e:
        print(f"Error loading word list: {e}")
        return
    
    # Choose mode
    while True:
        print("\nChoose mode:")
        print("1. Interactive solver (I pick a word, you guess it)")
        print("2. Auto-solve test (solver plays against a specific answer)")
        mode = input("Enter choice (1 or 2): ").strip()
        
        if mode == '1':
            solver.solve_interactive()
            break
        elif mode == '2':
            # Test with a specific answer
            test_answer = input("Enter answer to test (or press enter for random): ").strip().upper()
            if not test_answer:
                test_answer = random.choice(solver.words)
                print(f"Testing with random answer: {test_answer}")
            elif test_answer not in solver.words:
                print(f"'{test_answer}' is not in the word list. Using random.")
                test_answer = random.choice(solver.words)
                print(f"Testing with random answer: {test_answer}")
            
            solver.solve_for_answer(test_answer)
            break
        else:
            print("Please enter 1 or 2")

if __name__ == "__main__":
    main()
