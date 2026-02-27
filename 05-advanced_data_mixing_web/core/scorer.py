import random
import math

class DataScorer:
    def __init__(self, method="heuristic"):
        self.method = method

    def score(self, data_item):
        """
        Calculate scores for a single data item.
        Returns a dict with 'quality', 'learnability', 'difficulty', and 'composite_score'.
        """
        text = data_item.get('instruction', '') + " " + data_item.get('input', '') + " " + data_item.get('output', '')
        
        if self.method == "heuristic":
            return self._heuristic_score(text)
        elif self.method == "random":
            return self._random_score()
        else:
            return self._random_score()

    def _heuristic_score(self, text):
        # 1. Difficulty: Approximated by length (longer = harder)
        length = len(text)
        # Normalize length score roughly to 0-1 (assuming max length ~2000 chars)
        difficulty = min(length / 2000.0, 1.0)
        
        # 2. Quality: Approximated by unique word ratio (higher diversity = higher quality)
        words = text.split()
        if not words:
            unique_ratio = 0
        else:
            unique_ratio = len(set(words)) / len(words)
        quality = unique_ratio
        
        # 3. Learnability: Simulated. 
        # Assumption: Moderate difficulty items have high learnability (inverted U-curve), 
        # but for simplicity, let's say shorter items are faster to learn initially.
        learnability = 1.0 - difficulty
        
        # Composite Score: Weighted average
        # User might want to sort by specific metrics, but we provide a default composite.
        # Let's assume we value high quality and moderate difficulty.
        composite = (quality * 0.4) + (difficulty * 0.3) + (learnability * 0.3)
        
        return {
            "quality": round(quality, 4),
            "difficulty": round(difficulty, 4),
            "learnability": round(learnability, 4),
            "composite_score": round(composite, 4)
        }

    def _random_score(self):
        return {
            "quality": round(random.random(), 4),
            "difficulty": round(random.random(), 4),
            "learnability": round(random.random(), 4),
            "composite_score": round(random.random(), 4)
        }

    def batch_score(self, data_list):
        """
        Score a list of data items.
        Injects score fields into the items directly.
        """
        scored_data = []
        for item in data_list:
            scores = self.score(item)
            new_item = item.copy()
            new_item['scores'] = scores
            scored_data.append(new_item)
        return scored_data
