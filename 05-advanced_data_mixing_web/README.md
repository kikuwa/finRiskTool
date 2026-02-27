# Advanced SFT Data Mixing & Ordering Web Tool

This tool implements advanced data mixing and ordering strategies inspired by Curriculum Learning and Data Efficacy research.

## Features

1.  **Data Scoring**:
    *   **Quality**: Estimated via token diversity (Unique Token Ratio).
    *   **Difficulty**: Estimated via sequence length (Length-based heuristic).
    *   **Learnability**: Simulated metric derived from difficulty.
    *   *(Extendable to use real Model Perplexity/Loss)*

2.  **Data Mixing**:
    *   Precise control over Positive/Negative sample ratio.
    *   Oversampling support for small datasets.

3.  **Data Ordering (Curriculum Learning)**:
    *   **Ascending**: Easy $\to$ Hard.
    *   **Descending**: Hard $\to$ Easy.
    *   **Folded / Cyclic**: Splits data into $K$ folds, each sorted Easy $\to$ Hard, then concatenated. This creates a "sawtooth" difficulty pattern, allowing the model to revisit easy concepts periodically while progressing.

## Usage

1.  Run the web app:
    ```bash
    streamlit run app.py
    ```
2.  Configure settings in the Sidebar.
3.  Click "Process Data" to generate, visualize, and download the dataset.

## Directory Structure

*   `app.py`: Main Streamlit application.
*   `core/`: Backend logic for scoring, sorting, and mixing.
    *   `scorer.py`: Computes data metrics.
    *   `sorter.py`: Implements curriculum sorting strategies.
    *   `mixer.py`: Orchestrates the pipeline.
