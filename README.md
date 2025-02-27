# SMT-Task
There is total 5 steps in the task. 
# Sentence Contradiction Classification

## Project Overview
This project aims to develop a machine learning model that classifies pairs of sentences into one of three categories based on their semantic relationships:
- **Contradiction**: The sentences have opposite meanings.
- **Entailment**: One sentence logically follows from the other.
- **Neutral**: The sentences are related but do not imply each other.

## Dataset Description
The dataset consists of labeled and unlabeled sentence pairs provided in CSV format:
- **train.csv** (Labeled Training Data)
  - `id`: Unique identifier for each sentence pair.
  - `sentence1`: The first sentence (Premise).
  - `sentence2`: The second sentence (Hypothesis).
  - `label`: Relationship classification:
    - `0` = Contradiction
    - `1` = Neutral
    - `2` = Entailment
- **test.csv** (Unlabeled data for prediction)

Dataset Link: [Click Here](https://drive.google.com/file/d/1xp14KlixH2PZwL0YO0JooahJQ0IgfXJf/view?usp=sharing)

## Implementation Steps
### 1. Exploratory Data Analysis (EDA)
- Visualize the distribution of `Contradiction`, `Entailment`, and `Neutral` labels.
- Analyze sentence structure (length, word distribution, common words).
- Identify missing values or outliers.

### 2. Text Preprocessing
- Tokenization: Split sentences into words.
- Lowercasing: Convert text to lowercase.
- Remove stop words, special characters, and punctuation.
- Stemming/Lemmatization: Normalize words to their root form.
- Feature Extraction: Convert text into numeric representations using:
  - TF-IDF
  - Word2Vec
  - Transformer embeddings (BERT, XLM-R)

### 3. Model Creation
- **Baseline Models:** Random Forest, Decision Trees, XGBoost.
- **Neural Networks:** Custom Artificial Neural Network (ANN).
- **Advanced Models:** LSTM/GRU for sequence-based learning.
- **Transformer-Based Models:** Fine-tuning BERT/XLM-R for contextual understanding.

### 4. Model Evaluation
- Compute classification metrics: Accuracy, Precision, Recall, F1-score.
- Plot Confusion Matrix to analyze misclassifications.
- Generate an AUC-ROC curve to evaluate classification performance.

### 5. Model Tuning and Optimization
- Experiment with different optimizers (Adam, SGD, etc.).
- Adjust learning rate, batch size, and number of epochs.
- Use Grid Search or Random Search for hyperparameter tuning.

## Steps to Run the Code
1. Clone the repository:
   ```sh
   git clone <repository_link>
   cd <repository_name>
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the EDA and preprocessing script:
   ```sh
   python eda_preprocessing.py
   ```
4. Train the model:
   ```sh
   python train_model.py
   ```
5. Evaluate the model:
   ```sh
   python evaluate_model.py
   ```


