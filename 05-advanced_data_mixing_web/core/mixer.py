import json
import random
import os
from .scorer import DataScorer
from .sorter import DataSorter

class DataMixer:
    def __init__(self):
        self.scorer = DataScorer() # Default heuristic
        self.sorter = DataSorter()

    def load_jsonl(self, filepath):
        data = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except:
                    pass
        return data

    def process(self, pos_path, neg_path, ratio, total_count=None, 
                scoring_method='heuristic', sort_strategy='random', 
                sort_key='composite_score', num_folds=3, oversample=False,
                pos_multiplier=1.0, neg_multiplier=1.0):
        
        # 1. Load Data
        pos_data = self.load_jsonl(pos_path)
        neg_data = self.load_jsonl(neg_path)
        
        # 2. Score Data (Needed before selection if we want to select by score, 
        # but usually we select then sort. However, if we want to visualize distribution,
        # we should score everything? Scoring everything might be slow if large.
        # For this tool, we assume datasets are manageable or we just score selected?
        # User wants "Data Efficacy... Data Ordering". 
        # Let's score ALL data first to allow potential "Top-K" selection later,
        # and definitely for sorting.)
        
        self.scorer.method = scoring_method
        pos_data = self.scorer.batch_score(pos_data)
        neg_data = self.scorer.batch_score(neg_data)
        
        pos_weight = ratio * pos_multiplier
        neg_weight = (1 - ratio) * neg_multiplier
        if pos_weight + neg_weight == 0:
            effective_ratio = ratio
        else:
            effective_ratio = pos_weight / (pos_weight + neg_weight)

        if total_count is None:
            if effective_ratio == 0:
                calculated_total = len(neg_data)
            elif effective_ratio == 1:
                calculated_total = len(pos_data)
            else:
                max_pos = len(pos_data) / effective_ratio
                max_neg = len(neg_data) / (1 - effective_ratio)
                calculated_total = int(min(max_pos, max_neg))
            target_pos = int(calculated_total * effective_ratio)
            target_neg = calculated_total - target_pos
        else:
            target_pos = int(total_count * effective_ratio)
            target_neg = total_count - target_pos
            
        # Sample
        # For now, use Random Sampling for selection, as "Data Efficacy" usually implies
        # using the score for filtering, but user asked for "Data Ordering" mainly.
        # We can add "Filter Low Quality" later.
        
        if target_pos > len(pos_data):
            if oversample:
                sampled_pos = random.choices(pos_data, k=target_pos)
            else:
                sampled_pos = pos_data # Cap at max? Or error? previous logic raised error.
                # Let's just take all if not strict
                sampled_pos = pos_data
        else:
            sampled_pos = random.sample(pos_data, target_pos)
            
        if target_neg > len(neg_data):
            if oversample:
                sampled_neg = random.choices(neg_data, k=target_neg)
            else:
                sampled_neg = neg_data
        else:
            sampled_neg = random.sample(neg_data, target_neg)
            
        mixed_data = sampled_pos + sampled_neg
        
        # 4. Sort
        # Sort needs to know the key.
        sorted_data = self.sorter.sort_data(mixed_data, strategy=sort_strategy, 
                                          key=sort_key, num_folds=num_folds)
        
        return sorted_data

    def save_jsonl(self, data, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
