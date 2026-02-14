# 🧠 Software Defect Prediction using Machine Learning & Deep Learning (PROMISE Datasets)

> A comparative study of software defect prediction using **two different representations of software**:
> 1) Software Engineering Metrics
> 2) Raw Source Code (NLP Based)

This project implements an end-to-end defect prediction pipeline using classical ML, ensemble learning, and deep learning models on two different versions of the PROMISE dataset.

---

# 📌 Project Motivation

Software testing consumes **40-60% of development cost**.  
If we can predict defect-prone modules before testing, companies can:

- Prioritize risky files
- Reduce testing effort
- Save cost
- Improve reliability
- Enable intelligent CI/CD pipelines

This project investigates:

> **Does software metrics predict bugs better — OR — does source code itself predict bugs better?**

---

# 🧪 Experimental Design

We conduct two independent experiments:

| Part | Dataset Type | Prediction Input |
|----|----|----|
| PART-1 | Structured Metrics | Complexity, LOC, Halstead metrics |
| PART-2 | Source Code | Raw Java code text |

---

# 📂 DATASET 1 — PROMISE SOFTWARE METRICS

Source:
https://github.com/ssea-lab/PROMISE

Contains 34 project CSV files with software engineering metrics.

---

## 🔧 Data Preprocessing

### Dataset Preparation
- Combined 34 CSV files into one dataset
- Removed non-informative column (`name`)
- Converted bug count → binary classification

```
bug > 0 → Defective (1)
bug = 0 → Clean (0)
```

### Feature Engineering
- Train Test Split (80/20)
- StandardScaler normalization

---

## 🤖 Models Implemented (Metrics Dataset)

### Classical ML Models
- Decision Tree
- Random Forest
- KNN

### Deep Learning Models
- ANN (Feedforward Neural Network)
- DNN (Deep Neural Network)

---

## 🔍 Hyperparameter Optimization

| Model | Optimization Method |
|----|----|
| Decision Tree | GridSearchCV |
| Random Forest | RandomizedSearchCV |
| ANN | Architecture tuning |
| DNN | Layer tuning |

---

## 📊 Result (Metrics Dataset)

> Best Model: **DNN**
> Accuracy ≈ **72%**

Conclusion:
Deep neural networks capture relationships between software metrics better than shallow models.

---

# 📂 DATASET 2 — SOURCE CODE BASED DEFECT PREDICTION

Source:
https://github.com/alrz1999/PROMISE-dataset-csv

Dataset contains:
- Java source code (SRC)
- Bug label

This dataset allows predicting defects **directly from code text**.

---

## 🧹 Code Cleaning

Raw code cannot be fed to ML models directly.

We cleaned code by:

- Removing comments
- Removing symbols
- Lowercasing
- Removing numbers
- Normalizing whitespace

---

## ✏️ Feature Extraction (NLP)

### TF-IDF Vectorization
```
max_features = 5000
ngram_range = (1,2)
```

This captures programming patterns like:
- null pointer
- try catch
- return null

---

## ⚖️ Class Imbalance Handling

Bug datasets are highly imbalanced.

We applied:

> SMOTE (Synthetic Minority Oversampling Technique)

To improve bug recall.

---

## 🤖 Models Implemented (Code Dataset)

### Machine Learning
- Random Forest
- Tuned Random Forest
- Linear SVM
- Calibrated SVM
- SMOTE + SVM

### Deep Learning (Sequence Based)
- Tokenization
- Padding sequences
- BiLSTM Neural Network

---

## 🧠 Deep Learning Pipeline

Instead of TF-IDF:

```
Code → Tokenizer → Sequences → Padding → BiLSTM
```

This allows model to understand program structure rather than word frequency.

---

## 🎯 Evaluation Metrics

We focus on defect detection ability:

- Accuracy
- Precision
- Recall (MOST IMPORTANT)
- F1 Score
- Confusion Matrix

---

# 🏆 Key Findings

| Dataset Type | Best Model | Insight |
|----|----|----|
| Metrics Based | DNN | Metrics contain predictive patterns |
| Source Code Based | BiLSTM | Code semantics stronger than metrics |

---

# 🛠️ Tech Stack

| Category | Tools |
|----|----|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML | Scikit-learn |
| NLP | TF-IDF, Tokenizer |
| Deep Learning | TensorFlow / Keras |
| Imbalance Handling | SMOTE |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

# ▶️ How to Run

## 1. Clone Repository
```bash
git clone https://github.com/<your-username>/software-defect-prediction.git
cd software-defect-prediction
```

## 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn
```

## 3. Run Notebook
```bash
jupyter notebook assignment.ipynb
```

---

# 💡 Real World Applications

- Smart testing prioritization
- CI/CD failure prediction
- Risk-aware code review
- Automated QA planning
- Technical debt estimation

---

# 👨‍💻 Author
**Shreyash Kashyap**  
B.Tech CSE

---

# 📜 License
Academic / Research Purpose Only
