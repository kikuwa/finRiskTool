class DataSorter:
    def __init__(self):
        pass

    def sort_data(self, data, strategy='ascending', key='composite_score', num_folds=3):
        """
        Sort data based on strategy and key score.
        Strategies:
        - random: Random shuffle
        - ascending: Easy to Hard (Low score to High score)
        - descending: Hard to Easy (High score to Low score)
        - folded: Multi-stage curriculum (Cyclic Easy->Hard)
        """
        # Ensure data has scores
        if not data or 'scores' not in data[0]:
            # Fallback if no scores
            if strategy == 'random':
                random.shuffle(data)
                return data
            else:
                # Can't sort without scores, return as is or error. 
                # Let's assume scorer is always run before sorter.
                return data

        if strategy == 'random':
            import random
            random.shuffle(data)
            return data

        # Base Sort
        # Assuming 'ascending' means low score to high score. 
        # If key is 'difficulty', ascending = easy to hard.
        reverse = (strategy == 'descending')
        
        sorted_data = sorted(data, key=lambda x: x['scores'].get(key, 0), reverse=reverse)

        if strategy in ['ascending', 'descending']:
            return sorted_data

        if strategy == 'folded':
            return self._folded_sort(sorted_data, num_folds)
        
        return sorted_data

    def _folded_sort(self, sorted_data, num_folds):
        """
        Implements a Cyclic Curriculum: Splitting sorted data into K buckets,
        and then concatenating them. 
        Wait, if data is already sorted [0, 1, ..., 100], splitting into K buckets 
        [0..33], [34..66], [67..100] and concatenating gives [0..100], which is just sorted.
        
        Folded Ordering usually implies we want to mix difficulties but maintain a structure.
        
        Interpretation 1 (Cyclic): We want [Easy, Hard, Easy, Hard].
        So we take [Easy1, Easy2, ...], [Hard1, Hard2, ...]
        This is not what sorted list gives.
        
        Let's implement "Interleaved Folded":
        Split sorted data into K chunks.
        Take 1st from Chunk1, 1st from Chunk2, ..., 1st from ChunkK.
        2nd from Chunk1, 2nd from Chunk2, ...
        
        Example: [0, 1, 2, 3, 4, 5], K=2
        Chunk1: [0, 1, 2], Chunk2: [3, 4, 5]
        Result: [0, 3, 1, 4, 2, 5]
        This mixes easy and hard uniformly, but maintaining local structure.
        
        Interpretation 2 (Multi-stage Curriculum / Cyclic):
        User said "Multi-stage curriculum loops".
        This implies we want to TRAIN on Easy->Hard, then Easy->Hard again.
        BUT, we are just outputting a static dataset file.
        If the trainer reads linearly, then [0..33, 34..66, 67..100] is just one big Easy->Hard.
        
        To achieve "Loops" in a static file, we need to DUPLICATE data or PARTITION data.
        If we just reorder:
        We distribute the data into K "epochs" within the dataset.
        Epoch 1 gets 1/K of the easiest, 1/K of medium, 1/K of hard? 
        No, that's just random/uniform.
        
        Let's try: Divide data into K partitions randomly? No, that breaks ordering.
        
        Correct interpretation of "Multi-stage Curriculum" in a static file:
        The dataset is organized such that the model sees a full curriculum (Easy->Hard) multiple times.
        But we only have N samples.
        So we split N samples into K groups.
        Group 1: A subset of data, sorted Easy->Hard.
        Group 2: Another subset, sorted Easy->Hard.
        ...
        
        Algorithm:
        1. Sort all data by difficulty: [d1, d2, ..., dN] (d1=easiest)
        2. Distribute these into K buckets in a Round-Robin fashion to ensure balanced difficulty distribution across buckets?
           No, if we RR, then Bucket 1 has [d1, d(1+K), ...], Bucket 2 has [d2, d(2+K), ...]
           All buckets have similar difficulty distributions (Full spectrum).
        3. Sort EACH bucket Easy->Hard.
        4. Concatenate Buckets.
        
        Result: Easy->Hard (subset 1) -> Easy->Hard (subset 2) -> ...
        This creates K "waves" of curriculum.
        """
        if num_folds < 1:
            num_folds = 1
            
        # Distribute into K buckets
        buckets = [[] for _ in range(num_folds)]
        for i, item in enumerate(sorted_data):
            buckets[i % num_folds].append(item)
            
        # Buckets are already sorted because source was sorted and we distributed RR?
        # Example: 0, 1, 2, 3. K=2.
        # B1: 0, 2. B2: 1, 3.
        # Both are sorted.
        
        # Concatenate buckets
        result = []
        for bucket in buckets:
            result.extend(bucket)
            
        return result
