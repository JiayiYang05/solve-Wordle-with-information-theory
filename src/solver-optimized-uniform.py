import requests
import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Set
import random
from numba import jit  # For JIT compilation
import pickle
import os
import time

class FastWordleSolver:
    def __init__(self, use_frequency=False, cache_dir="wordle_cache"):
        self.words = self.get_wordle_answers()
        self.use_frequency = use_frequency
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try to load precomputed data, otherwise compute it
        self.pattern_matrix, self.word_to_idx, self.idx_to_word = self.load_or_compute_patterns()
        
        print(f"✅ Solver initialized with {len(self.words)} words")
    
    def get_cache_files(self):
        """Return paths to cache files."""
        return {
            'matrix': os.path.join(self.cache_dir, 'pattern_matrix.npy'),
            'mapping': os.path.join(self.cache_dir, 'word_mapping.pkl'),
            'words': os.path.join(self.cache_dir, 'word_list.pkl')
        }
    
    def should_recompute(self):
        """Check if we need to recompute the pattern matrix."""
        cache_files = self.get_cache_files()
        
        # Check if all cache files exist
        if not all(os.path.exists(path) for path in cache_files.values()):
            return True
        
        # Check if word list has changed
        try:
            with open(cache_files['words'], 'rb') as f:
                cached_words = pickle.load(f)
            if cached_words != self.words:
                print("📝 Word list has changed, recomputing patterns...")
                return True
        except:
            return True
        
        return False
    
    def load_or_compute_patterns(self):
        """Load precomputed patterns or compute them if needed."""
        cache_files = self.get_cache_files()
        
        if not self.should_recompute():
            print("📂 Loading precomputed patterns from cache...")
            try:
                pattern_matrix = np.load(cache_files['matrix'])
                with open(cache_files['mapping'], 'rb') as f:
                    data = pickle.load(f)
                    word_to_idx = data['word_to_idx']
                    idx_to_word = data['idx_to_word']
                
                print("✅ Precomputed patterns loaded successfully!")
                return pattern_matrix, word_to_idx, idx_to_word
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}, recomputing...")
        
        # Compute patterns and save to cache
        print("🔄 Precomputing patterns (this may take 1-2 minutes)...")
        start_time = time.time()
        
        pattern_matrix = self.precompute_pattern_matrix()
        word_to_idx = {word: i for i, word in enumerate(self.words)}
        idx_to_word = {i: word for i, word in enumerate(self.words)}
        
        # Save to cache
        self.save_to_cache(pattern_matrix, word_to_idx, idx_to_word)
        
        compute_time = time.time() - start_time
        print(f"✅ Pattern computation completed in {compute_time:.2f} seconds")
        
        return pattern_matrix, word_to_idx, idx_to_word
    
    def save_to_cache(self, pattern_matrix, word_to_idx, idx_to_word):
        """Save precomputed data to cache files."""
        cache_files = self.get_cache_files()
        
        try:
            # Save pattern matrix as numpy file
            np.save(cache_files['matrix'], pattern_matrix)
            
            # Save mappings as pickle
            with open(cache_files['mapping'], 'wb') as f:
                pickle.dump({
                    'word_to_idx': word_to_idx,
                    'idx_to_word': idx_to_word
                }, f)
            
            # Save word list to detect changes
            with open(cache_files['words'], 'wb') as f:
                pickle.dump(self.words, f)
            
            print("💾 Precomputed patterns saved to cache")
        except Exception as e:
            print(f"⚠️ Warning: Could not save cache: {e}")

   
    def get_wordle_answers(self) -> List[str]:
        """Get the full list of Wordle answers."""
        url = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"
        response = requests.get(url)
        words = [word.strip().upper() for word in response.text.split('\n') if len(word.strip()) == 5]
        return words
    
    def pattern_to_int(self, pattern: str) -> int:
        """Convert pattern string to integer for faster processing."""
        # G=2, Y=1, X=0, so pattern becomes a base-3 number
        mapping = {'G': 2, 'Y': 1, 'X': 0}
        result = 0
        for i, char in enumerate(pattern):
            result += mapping[char] * (3 ** (4 - i))  # Most significant digit first
        return result
    
    def int_to_pattern(self, pattern_int: int) -> str:
        """Convert integer back to pattern string."""
        mapping = {2: 'G', 1: 'Y', 0: 'X'}
        pattern = []
        for i in range(5):
            digit = (pattern_int // (3 ** (4 - i))) % 3
            pattern.append(mapping[digit])
        return ''.join(pattern)
    
    def get_pattern_fast(self, guess_idx: int, answer_idx: int) -> int:
        """Get precomputed pattern as integer."""
        return self.pattern_matrix[guess_idx, answer_idx]
    
    def precompute_pattern_matrix(self) -> np.ndarray:
        """Precompute all patterns between all words as integers."""
        n = len(self.words)
        matrix = np.zeros((n, n), dtype=np.int32)
        
        print("Precomputing pattern matrix... (this may take a minute)")
        for i, guess in enumerate(self.words):
            for j, answer in enumerate(self.words):
                pattern_str = self.get_pattern_slow(guess, answer)  # Use slow version only once
                matrix[i, j] = self.pattern_to_int(pattern_str)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{n} words...")
        
        print("Pattern matrix precomputation complete!")
        return matrix
    
    def get_pattern_slow(self, guess: str, answer: str) -> str:
        """Original pattern calculation (used only for precomputation)."""
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
        
        return ''.join(pattern)
    
    def filter_words_fast(self, possible_indices: np.ndarray, guess_idx: int, pattern_int: int) -> np.ndarray:
        """Filter possible words using vectorized operations."""
        # Get patterns for this guess against all possible answers
        patterns = self.pattern_matrix[guess_idx, possible_indices]
        # Keep only words that match the pattern
        return possible_indices[patterns == pattern_int]
    
    def calculate_entropy_fast(self, guess_idx: int, possible_indices: np.ndarray) -> float:
        """Calculate entropy using vectorized operations."""
        if len(possible_indices) == 0:
            return 0
        
        # Get all patterns for this guess against possible answers
        patterns = self.pattern_matrix[guess_idx, possible_indices]
        
        # Count occurrences of each pattern using numpy
        unique_patterns, counts = np.unique(patterns, return_counts=True)
        
        # Calculate entropy
        probabilities = counts / len(possible_indices)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def get_best_guesses_fast(self, possible_indices: np.ndarray, top_n: int = 10, max_candidates: int = 200) -> List[Tuple[str, float]]:
        """Get top guesses using optimized entropy calculation."""
        n_possible = len(possible_indices)
        
        if n_possible <= 2:
            return [(self.idx_to_word[idx], math.log2(n_possible)) for idx in possible_indices]
        
        # Select candidate guesses intelligently
        # candidate_indices = self.select_candidate_guesses(possible_indices, max_candidates)
        candidate_indices =possible_indices
        
        # Calculate entropy for each candidate
        word_scores = []
        for candidate_idx in candidate_indices:
            entropy = self.calculate_entropy_fast(candidate_idx, possible_indices)
            word_scores.append((self.idx_to_word[candidate_idx], entropy))
        
        # Sort and return top N
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_n]

    def clear_cache(self):
        """Clear the cache files (useful if you want to force recomputation)."""
        cache_files = self.get_cache_files()
        for file_path in cache_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        print("🗑️ Cache cleared")
    
    def solve_interactive_optimized(self):
        """Optimized interactive solver."""
        # Convert to indices for faster processing
        possible_indices = np.arange(len(self.words))
        answer_idx = random.randint(0, len(self.words) - 1)
        answer = self.idx_to_word[answer_idx]
        guesses_made = []
        
        print("🚀 Optimized Wordle Solver - Interactive Mode")
        print(f"Total possible answers: {len(self.words)}")
        print("Patterns will be shown automatically (G=Green, Y=Yellow, X=Grey)")
        
        for attempt in range(6):
            print(f"\n--- Attempt {attempt + 1} ---")
            print(f"Remaining possible answers: {len(possible_indices)}")
            
            if len(possible_indices) <= 10:
                words = [self.idx_to_word[idx] for idx in possible_indices]
                print(f"Possible answers: {', '.join(words)}")
            
            print("Calculating best guesses...")
            
            # Get best guesses
            best_guesses = self.get_best_guesses_fast(possible_indices)
            
            print("\nTop recommendations:")
            for i, (word, entropy) in enumerate(best_guesses[:8]):
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
                elif guess not in self.word_to_idx:
                    print(f"'{guess}' is not in the word list!")
                    continue
                else:
                    break
            
            guess_idx = self.word_to_idx[guess]
            pattern_int = self.pattern_matrix[guess_idx, answer_idx]
            pattern = self.int_to_pattern(pattern_int)
            guesses_made.append((guess, pattern))
            
            print(f"Pattern: {pattern}")
            self.display_colored_guess(guess, pattern)
            
            # Filter possible answers
            previous_count = len(possible_indices)
            possible_indices = self.filter_words_fast(possible_indices, guess_idx, pattern_int)
            
            print(f"Filtered from {previous_count} to {len(possible_indices)} possible answers")
            
            # Check if solved
            if pattern == 'GGGGG':
                print(f"\n🎉 Congratulations! You solved it in {attempt + 1} attempts!")
                print(f"The answer was: {answer}")
                self.display_statistics(guesses_made, answer)
                input("\nPress Enter to return to main menu...")
                return
            elif len(possible_indices) == 1:
                remaining_word = self.idx_to_word[possible_indices[0]]
                print(f"\nOnly one possibility remains: {remaining_word}")
        
        print(f"\nGame over! The answer was: {answer}")
        input("\nPress Enter to return to main menu...")
        # if len(possible_indices) <= 20:
        #     words = [self.idx_to_word[idx] for idx in possible_indices]
        #     print(f"Remaining possible answers: {', '.join(words)}")
    
    def display_colored_guess(self, guess: str, pattern: str):
        """Display the guess with colored output."""
        colors = {'G': '🟩', 'Y': '🟨', 'X': '⬜'}
        colored_output = ''.join(colors[char] for char in pattern)
        print(f"Visual: {colored_output}")
        
        print("Letters: ", end="")
        for letter, color in zip(guess, pattern):
            if color == 'G':
                print(f"\033[92m{letter}\033[0m", end=" ")
            elif color == 'Y':
                print(f"\033[93m{letter}\033[0m", end=" ")
            else:
                print(f"\033[90m{letter}\033[0m", end=" ")
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

def main():
    print("🚀 Optimized Wordle Solver using Information Theory")
    print("=" * 50)   
    

    # Choose version
    while True:
        version = input("Choose version (1 for uniform distribution, 2 for frequency-based): ").strip()
        if version in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    use_frequency = (version == "2")

    # Initialize solver ONCE
    print("Initializing Optimized Wordle Solver...")
    try:
        solver = FastWordleSolver(use_frequency=use_frequency)
        print("✅ Solver initialized successfully!")
    except Exception as e:
        print(f"Error loading word list: {e}")
        return
       
    # Main game loop - keep playing until user quits
    while True:
        print("\n" + "="*50)
        print("🎮 MAIN MENU")
        print("1. Play a new game")
        print("2. Performance test (solve multiple words to test speed)")
        print("3. Quit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            solver.solve_interactive_optimized()
        elif choice == '2':
            test_performance(solver)
        elif choice == '3':
            print("Thanks for playing! Goodbye! 👋")
            break
        else:
            print("Please enter 1, 2, or 3")

def test_performance(solver):
    """Test the performance of the optimized solver."""
    print("\n🧪 Performance Test")
    print("Solving 10 random words to test speed...")
    
    test_words = random.sample(solver.words, min(10, len(solver.words)))
    total_attempts = 0
    start_time = time.time()
    
    for i, answer in enumerate(test_words):
        print(f"\nTest {i+1}/10: {answer}")
        possible_indices = np.arange(len(solver.words))
        answer_idx = solver.word_to_idx[answer]
        attempts = 0
        
        for attempt in range(6):
            attempts += 1
            best_guesses = solver.get_best_guesses_fast(possible_indices, top_n=1)
            guess = best_guesses[0][0]
            guess_idx = solver.word_to_idx[guess]
            pattern_int = solver.pattern_matrix[guess_idx, answer_idx]
            pattern = solver.int_to_pattern(pattern_int)
            
            if pattern == 'GGGGG':
                print(f"  Solved in {attempts} attempts: {guess}")
                total_attempts += attempts
                break
            
            possible_indices = solver.filter_words_fast(possible_indices, guess_idx, pattern_int)
        else:
            print(f"  Failed to solve in 6 attempts")
            total_attempts += 6
    
    end_time = time.time()
    avg_attempts = total_attempts / len(test_words)
    total_time = end_time - start_time
    
    print(f"\n📊 Performance Results:")
    print(f"Average attempts per word: {avg_attempts:.2f}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Time per word: {total_time/len(test_words):.2f} seconds")

    input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    import time
    main()
