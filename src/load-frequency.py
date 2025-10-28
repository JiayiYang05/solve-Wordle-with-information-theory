import gzip
import requests
import io
import pandas as pd
from collections import defaultdict
import os
import numpy as np


class FastWordleSolver:
    def __init__(self, use_frequency=True, cache_dir="wordle_cache",
                 frequency_source="bnc", frequency_file=None, normalization="sigmoid"):
        self.words = self.get_wordle_answers()
        self.use_frequency = use_frequency
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Try to load precomputed data, otherwise compute it
        self.pattern_matrix, self.word_to_idx, self.idx_to_word = self.load_or_compute_patterns()

        print(f"✅ Solver initialized with {len(self.words)} words")

        if use_frequency:
            if frequency_source == "bnc":
                self.word_frequencies = self.load_bnc_frequencies_compressed(normalization=normalization)
                # self.test_bnc_loading()
            elif frequency_source == "file" and frequency_file:
                self.word_frequencies = self.load_word_frequencies_from_file(frequency_file)
            else:
                self.word_frequencies = self.load_fallback_frequencies()

            print(f"✅ Using frequency-weighted solver ({frequency_source})")
        else:
            self.word_frequencies = {word: 1.0 for word in self.words}
            print("✅ Using uniform distribution solver")

    def get_wordle_answers(self) -> List[str]:
        """Get the full list of Wordle answers."""
        url = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"
        response = requests.get(url)
        words = [word.strip().upper() for word in response.text.split('\n') if len(word.strip()) == 5]
        return words
    
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



    def load_bnc_frequencies_compressed(self, bnc_url: str = "https://www.kilgarriff.co.uk/BNClists/lemma.al.gz",
                                      normalization: str = "sigmoid") -> Dict[str, float]:
        """Load and filter BNC frequencies from compressed file."""

        print("📊 Loading British National Corpus frequencies from compressed file...")

        try:
            # Download the compressed file
            response = requests.get(bnc_url)
            response.raise_for_status()

            # Decompress and parse
            with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
                bnc_data = f.readlines()

            print(f"✅ Decompressed BNC data, {len(bnc_data)} lines")

            # Parse BNC data (format: sort-order, frequency, word, word-class)
            bnc_freq = {}
            for line_num, line in enumerate(bnc_data):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            # parts[0] = sort-order, parts[1] = frequency, parts[2] = word, parts[3] = word-class
                            word = parts[2].upper().strip()
                            frequency = float(parts[1])
                            bnc_freq[word] = frequency
                        except (ValueError, IndexError) as e:
                            continue

            print(f"✅ Parsed {len(bnc_freq)} words from BNC")

            # Show first 10 words for testing
            print("\n🔍 First 20 words from BNC:")
            first_10 = list(bnc_freq.items())[:10]
            for i, (word, freq) in enumerate(first_10):
                print(f"  {i+1:2d}. {word:15} {freq:>10,}")

        except Exception as e:
            print(f"❌ Error loading BNC data: {e}")
            print("🔄 Using built-in fallback frequencies...")
            return self.load_fallback_frequencies()

        # Filter for Wordle words and handle missing words
        wordle_frequencies = {}
        missing_words = []

        for word in self.words:
            if word in bnc_freq:
                wordle_frequencies[word] = bnc_freq[word]
            else:
                missing_words.append(word)
                # Use frequency of 1 for missing words (very rare)
                wordle_frequencies[word] = 1.0

        if missing_words:
            print(f"⚠️  {len(missing_words)} Wordle words not found in BNC, assigned minimum frequency")
            if len(missing_words) <= 10:
                print(f"   Missing: {', '.join(missing_words)}")

        # Show most common Wordle words
        sorted_words = sorted(wordle_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🏆 Top 10 most common Wordle words in BNC:")
        for i, (word, freq) in enumerate(sorted_words):
            print(f"  {i+1:2d}. {word:15} {freq:>10,.0f}")

        # Apply normalization
        wordle_frequencies = self.apply_normalization(wordle_frequencies, normalization)

        # Show most common Wordle words normalized
        print(f"✅ Final frequency list: {len(wordle_frequencies)} words with {normalization} normalization")
        sorted_words = sorted(wordle_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🏆 Top 10 most common Wordle words in BNC:")
        for i, (word, freq) in enumerate(sorted_words):
            print(f"  {i+1:2d}. {word:15} {freq:>10,.3f}")
        return wordle_frequencies

    def apply_normalization(self, frequencies: Dict[str, float], method: str) -> Dict[str, float]:
        """Apply the chosen normalization method."""
        if method == "sigmoid":
            return self.normalize_frequencies_sigmoid(frequencies)
        elif method == "log":
            return self.normalize_frequencies_log_scale(frequencies)
        elif method == "softmax":
            return self.normalize_frequencies_softmax(frequencies)
        elif method == "rank":
            return self.normalize_frequencies_rank_based(frequencies)
        else:
            # Default min-max scaling
            freq_values = np.array(list(frequencies.values()))
            if np.max(freq_values) == np.min(freq_values):
                # All frequencies are the same
                normalized = np.ones_like(freq_values) * 0.5
            else:
                normalized = (freq_values - np.min(freq_values)) / (np.max(freq_values) - np.min(freq_values))
            words = list(frequencies.keys())
            return dict(zip(words, normalized))

    # Test function to verify the loading works
    def test_bnc_loading(self):
        """Test function to verify BNC loading works correctly."""
        print("🧪 Testing BNC frequency loading...")

        # Load a small subset of words for testing
        test_words = self.words[:50]  # First 50 Wordle words

        # Create a mock frequency function for testing
        test_frequencies = self.load_bnc_frequencies_compressed()

        # Show most common Wordle words
        sorted_words = sorted(test_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🏆 Top 10 most common Wordle words in BNC:")
        for i, (word, freq) in enumerate(sorted_words):
            print(f"  {i+1:2d}. {word:15} {freq:>10,.0f}")

    def normalize_frequencies_sigmoid(self, frequencies: Dict[str, float]) -> Dict[str, float]:
        """Normalize frequencies using sigmoid function to [0,1] range."""
        freq_values = np.array(list(frequencies.values()))
        
        # Log transform first to handle exponential distribution of word frequencies
        log_freq = np.log1p(freq_values)  # log(1 + x) to handle zeros
        
        # Sigmoid normalization
        # Center the data around 0 for better sigmoid behavior
        mean_log = np.mean(log_freq)
        std_log = np.std(log_freq)
        centered_log = (log_freq - mean_log) / std_log
        
        # Apply sigmoid
        normalized = 1 / (1 + np.exp(-centered_log))
        
        # Create normalized dictionary
        words = list(frequencies.keys())
        return dict(zip(words, normalized))

    def normalize_frequencies_log_scale(self, frequencies: Dict[str, float]) -> Dict[str, float]:
        """Normalize frequencies using log scaling."""
        freq_values = np.array(list(frequencies.values()))
        
        # Log transform and scale to [0,1]
        log_freq = np.log1p(freq_values)
        normalized = (log_freq - np.min(log_freq)) / (np.max(log_freq) - np.min(log_freq))
        
        words = list(frequencies.keys())
        return dict(zip(words, normalized))

    def normalize_frequencies_softmax(self, frequencies: Dict[str, float]) -> Dict[str, float]:
        """Normalize frequencies using softmax (turns them into probabilities)."""
        freq_values = np.array(list(frequencies.values()))
        
        # Softmax normalization
        exp_freq = np.exp(freq_values - np.max(freq_values))  # Subtract max for numerical stability
        softmax_freq = exp_freq / np.sum(exp_freq)
        
        words = list(frequencies.keys())
        return dict(zip(words, softmax_freq))

    def normalize_frequencies_rank_based(self, frequencies: Dict[str, float]) -> Dict[str, float]:
        """Normalize frequencies using rank-based approach (robust to outliers)."""
        freq_values = np.array(list(frequencies.values()))

        # Convert to ranks and normalize
        ranks = np.argsort(np.argsort(freq_values)) + 1  # Get ranks (1-based)
        normalized = ranks / len(ranks)  # Scale to [0,1]
        
        words = list(frequencies.keys())
        return dict(zip(words, normalized))

    # Updated entropy calculation for normalized frequencies:
    def calculate_entropy_weighted(self, guess: str, possible_answers: List[str]) -> float:
        """Calculate entropy using normalized frequency-weighted probabilities."""
        pattern_weights = {}
        total_weight = 0
        
        for answer in possible_answers:
            pattern = self.get_pattern(guess, answer)
            weight = self.word_frequencies.get(answer, 0.001)  # Small epsilon for stability
            
            pattern_weights[pattern] = pattern_weights.get(pattern, 0.0) + weight
            total_weight += weight
        
        if total_weight <= 0:
            return 0
        
        entropy = 0
        for pattern, weight in pattern_weights.items():
            p = weight / total_weight
            # Add small epsilon to avoid log(0)
            entropy -= p * math.log2(p + 1e-10)
        
        return entropy


if __name__ == "__main__":
    print("Initializing Optimized Wordle Solver...")
    try:
        solver = FastWordleSolver(normalization="sigmoid")
        print("✅ Solver initialized successfully!")
    except Exception as e:
        print(f"Error loading word list: {e}")
