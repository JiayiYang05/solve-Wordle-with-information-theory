import requests
import math
from collections import Counter
from typing import List, Tuple, Dict
import random
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import time

class WordleSolverGUI:
    def __init__(self, use_frequency=False):
        self.words = self.get_wordle_answers()
        self.use_frequency = use_frequency
        self.word_frequencies = {word: 1 for word in self.words}
        self.pattern_cache = {}
        
        # Game state
        self.answer = None
        self.possible_answers = None
        self.guesses_made = []
        self.current_attempt = 0
        self.game_over = False
        
        # Initialize GUI components
        self.setup_gui()
        
    def get_wordle_answers(self) -> List[str]:
        """Get a smaller subset of Wordle answers for testing."""
        url = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"
        response = requests.get(url)
        all_words = [word.strip().upper() for word in response.text.split('\n') if len(word.strip()) == 5]
        return all_words[:100]
    
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
        
        if n_answers <= 2:
            return [(word, math.log2(n_answers)) for word in possible_answers]
        
        # FIX: Only calculate entropy for words in possible_answers
        word_scores = []
        for word in possible_answers[:min(50, len(possible_answers))]:  # Limit for performance
            entropy = self.calculate_entropy_for_guess(word, possible_answers)
            word_scores.append((word, entropy))
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_n]
    
    def setup_gui(self):
        """Setup all GUI components."""
        # Title
        self.title = widgets.HTML(
            value="<h1 style='text-align: center; color: #6aaa64;'>🎯 Wordle Solver</h1>"
        )
        
        # Game info
        self.info_label = widgets.HTML(value="<h3>Game Information</h3>")
        self.remaining_label = widgets.Label(value="Remaining possible answers: --")
        self.attempt_label = widgets.Label(value="Attempt: 0/6")
        
        # Guess input - ADDED: on_submit callback for Enter key
        self.guess_input = widgets.Text(
            value='',
            placeholder='Enter your 5-letter guess',
            description='Guess:',
            disabled=False,
            layout=widgets.Layout(width='200px')
        )
        self.guess_input.observe(self.on_enter_key, 'value')
        
        # Buttons
        self.submit_button = widgets.Button(
            description="Submit Guess",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.auto_button = widgets.Button(
            description="Auto Guess",
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.new_game_button = widgets.Button(
            description="New Game",
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.reveal_button = widgets.Button(
            description="Reveal Answer",
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Button callbacks
        self.submit_button.on_click(self.on_submit)
        self.auto_button.on_click(self.on_auto)
        self.new_game_button.on_click(self.on_new_game)
        self.reveal_button.on_click(self.on_reveal)
        
        # Results display
        self.results_output = widgets.Output()
        self.recommendations_output = widgets.Output()
        
        # Game board (visual representation)
        self.board_output = widgets.Output()
        
        # Layout
        self.control_box = widgets.HBox([
            self.guess_input, 
            self.submit_button, 
            self.auto_button,
            self.new_game_button,
            self.reveal_button
        ], layout=widgets.Layout(justify_content='space-around'))
        
        self.info_box = widgets.VBox([
            self.info_label,
            self.remaining_label,
            self.attempt_label
        ], layout=widgets.Layout(width='300px'))
        
        # Main layout
        self.main_layout = widgets.VBox([
            self.title,
            widgets.HBox([self.info_box, self.board_output]),
            self.control_box,
            self.recommendations_output,
            self.results_output
        ])
    
    def on_enter_key(self, change):
        """Handle Enter key press in the guess input."""
        if change['name'] == 'value' and change['new']:
            # Check if the user pressed Enter (input ends with newline)
            if change['new'].endswith('\n'):
                # Remove the newline and process the guess
                current_value = change['new'].strip()
                self.guess_input.value = current_value
                self.on_submit(None)
    
    def display_colored_guess(self, guess: str, pattern: str):
        """Display colored boxes for the guess."""
        colors = {
            'G': '🟩',  # Green
            'Y': '🟨',  # Yellow  
            'X': '⬜'   # Grey
        }
        
        colored_boxes = ''.join(colors[char] for char in pattern)
        
        with self.board_output:
            display(HTML(f"""
            <div style='font-family: monospace; font-size: 24px; margin: 10px;'>
                <strong>{guess}</strong><br>
                {colored_boxes}
            </div>
            """))
    
    def update_recommendations(self):
        """Update the recommendations display."""
        with self.recommendations_output:
            clear_output()
            if self.game_over:
                return
                
            print("🔄 Calculating best guesses...")
            best_guesses = self.get_best_guesses(self.possible_answers)
            
            print("\n🏆 Top Recommendations:")
            for i, (word, entropy) in enumerate(best_guesses[:8]):
                print(f"{i+1:2d}. {word} (entropy: {entropy:.3f})")
    
    def on_submit(self, button):
        """Handle submit button click or Enter key press."""
        if self.game_over:
            return
            
        guess = self.guess_input.value.strip().upper()  # Convert to uppercase
        
        if len(guess) != 5:
            with self.results_output:
                clear_output()
                print("❌ Guess must be exactly 5 letters!")
            return
        
        # FIX: Check if guess is in possible_answers, not just self.words
        if guess not in self.possible_answers:
            with self.results_output:
                clear_output()
                print(f"❌ '{guess}' is not in the remaining possible answers!")
                print("💡 Try one of the recommendations above.")
            return
        
        # Process the guess
        pattern = self.get_pattern(guess, self.answer)
        self.guesses_made.append((guess, pattern))
        self.current_attempt += 1
        
        # Update display
        self.display_colored_guess(guess, pattern)
        self.possible_answers = self.filter_words(self.possible_answers, guess, pattern)
        
        # Update labels
        self.remaining_label.value = f"Remaining possible answers: {len(self.possible_answers)}"
        self.attempt_label.value = f"Attempt: {self.current_attempt}/6"
        
        # Clear input
        self.guess_input.value = ''
        
        # Check game state
        with self.results_output:
            clear_output()
            if pattern == 'GGGGG':
                self.game_over = True
                print(f"🎉 Congratulations! You solved it in {self.current_attempt} attempts!")
                print(f"🏆 The answer was: {self.answer}")
                self.display_statistics()
            elif self.current_attempt >= 6:
                self.game_over = True
                print(f"💔 Game Over! The answer was: {self.answer}")
                if len(self.possible_answers) <= 10:
                    print(f"Remaining possibilities: {', '.join(self.possible_answers)}")
            else:
                print(f"🔍 Filtered to {len(self.possible_answers)} possible answers")
        
        # Update recommendations
        if not self.game_over:
            self.update_recommendations()
    
    def on_auto(self, button):
        """Handle auto guess button click."""
        if self.game_over:
            return
            
        best_guesses = self.get_best_guesses(self.possible_answers, top_n=1)
        if best_guesses:
            # Auto-fill the input but don't submit automatically
            self.guess_input.value = best_guesses[0][0]
            # User can now see the word and press Enter or click Submit
    
    def on_new_game(self, button):
        """Start a new game."""
        self.answer = random.choice(self.words)
        self.possible_answers = self.words.copy()
        self.guesses_made = []
        self.current_attempt = 0
        self.game_over = False
        
        # Clear displays
        with self.board_output:
            clear_output()
        with self.results_output:
            clear_output()
        
        # Update labels
        self.remaining_label.value = f"Remaining possible answers: {len(self.possible_answers)}"
        self.attempt_label.value = f"Attempt: {self.current_attempt}/6"
        
        # Show initial state
        with self.results_output:
            print("🎮 New game started!")
            print(f"🤫 I'm thinking of a 5-letter word...")
        
        # Show initial recommendations
        self.update_recommendations()
    
    def on_reveal(self, button):
        """Reveal the answer."""
        if self.answer:
            with self.results_output:
                clear_output()
                print(f"🤫 The answer is: {self.answer}")
    
    def display_statistics(self):
        """Display game statistics."""
        with self.results_output:
            print(f"\n📊 Statistics:")
            print(f"Solved in {len(self.guesses_made)} attempts")
            
            print(f"\n📝 Guess history:")
            for i, (guess, pattern) in enumerate(self.guesses_made, 1):
                colored = ''.join(['🟩' if c == 'G' else '🟨' if c == 'Y' else '⬜' for c in pattern])
                print(f"Attempt {i}: {guess} -> {pattern} {colored}")
            
            initial_entropy = math.log2(len(self.words))
            print(f"\n🧠 Information theory:")
            print(f"Initial entropy: {initial_entropy:.2f} bits")
            if len(self.guesses_made) > 0:
                print(f"Information gained per guess: {initial_entropy/len(self.guesses_made):.2f} bits")
    
    def show(self):
        """Display the GUI and start a new game."""
        display(self.main_layout)
        self.on_new_game(None)  # Start first game

# Additional function to run in Colab
def run_wordle_solver(use_frequency=False):
    """Run the Wordle Solver GUI in Colab."""
    print("🚀 Starting Wordle Solver GUI...")
    solver_gui = WordleSolverGUI(use_frequency=use_frequency)
    solver_gui.show()
    return solver_gui

# Demo function with example
def demo_wordle_solver():
    """Run a demo of the Wordle Solver."""
    print("""
    🎯 Wordle Solver Demo
    ====================
    
    How to use:
    1. Enter your 5-letter guess in the text box (lowercase or uppercase)
    2. Click "Submit Guess" OR press Enter to submit
    3. Use "Auto Guess" to see the solver's recommendation
    4. The colored boxes show: 🟩=Green, 🟨=Yellow, ⬜=Grey
    5. Recommendations are updated after each guess
    
    The solver uses information theory to maximize entropy and eliminate possibilities!
    """)
    
    # Create and display the solver
    solver = WordleSolverGUI()
    solver.show()
    
    return solver

# If running directly in Colab, start the demo
if __name__ == "__main__":
    # This will automatically run when the cell is executed in Colab
    demo = demo_wordle_solver()
