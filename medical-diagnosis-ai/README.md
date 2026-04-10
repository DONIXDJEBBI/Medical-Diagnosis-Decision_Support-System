# Medical Diagnosis Decision Support System
## Comparative Analysis: Fuzzy Logic vs Decision Tree

**Student**: Houssem Eddine Djebbi (ID: 2430210117)  
**Course**: AIN7101 Master Project  
**Date**: January 2026

---

## 1. Project Overview

This project implements and compares two distinct approaches to medical diagnosis support:

- **Fuzzy Logic**: Rule-based system emphasizing interpretability
- **Decision Tree**: Machine learning classifier emphasizing accuracy

Both models diagnose four diseases (Flu, Pneumonia, Allergy, Bronchitis) based on nine medical features (5 symptoms + 4 lab tests).

### Key Results

| Metric | Fuzzy Logic | Decision Tree | Winner |
|--------|------------|---------------|--------|
| **Accuracy** | 35.5% | 56.75% | Decision Tree |
| **F1-Score** | 0.2939 | 0.4785 | Decision Tree |
| **Inference Time (ms/sample)** | 2-5 | 1-2 | Decision Tree |
| **Memory Usage** | ~5 MB | ~2 MB | Decision Tree |
| **Interpretability** | 100% | ~60% | Fuzzy Logic |

---

## 2. Project Structure

```
medical-diagnosis-ai/
|-- src/
|   |-- fuzzy_diagnosis.py       # Fuzzy Logic inference engine
|   |-- ml_decision_tree.py      # ML classifier pipeline
|   |-- evaluate.py              # Model evaluation utilities
|   |-- compare.py               # Comprehensive comparison script
|   |-- generate_data.py         # Synthetic data generation
|   `-- app.py                   # Streamlit web interface
|-- data/
|   `-- synthetic_patients.csv   # 1,200 patient records
|-- results/
|   |-- metrics.json             # Performance metrics
|   `-- comparison.json          # Detailed comparison results
|-- main.py                      # Entry point for all commands
|-- requirements.txt             # Python dependencies
`-- README.md                    # This file
```

---

## 3. Installation

### Step 1: Extract Project
```bash
cd medical-diagnosis-ai
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the System

### 4.1 Main Commands (Recommended)

**Run Full Comparison**:
```bash
python main.py --mode compare
```

**Run Fuzzy Logic Only**:
```bash
python main.py --model fuzzy
```

**Run Decision Tree Only**:
```bash
python main.py --model tree
```

**Run with custom tree parameters**:
```bash
python main.py --mode compare --max-depth 4 --min-samples-leaf 8
```

### 4.2 Direct Comparison Script

**Detailed comparison with all metrics**:
```bash
python src/compare.py
```

**Quick demo runner (one-shot)**:
```bash
python demo.py
```

### 4.3 Alternative Commands

**Model evaluation only**:
```bash
python src/evaluate.py
```
Includes 5-fold stratified cross-validation by default and writes results to `results/metrics.json`.

**Interactive web interface**:
```bash
streamlit run src/app.py
```

**Generate fresh data**:
```bash
python src/generate_data.py
```

---

## 5. Expected Output

### Running `python main.py --mode compare`

**Console Output Sample**:
```
======================================================================
COMPARING FUZZY LOGIC vs DECISION TREE
======================================================================

Loading dataset...
[OK] Loaded 1200 patient records

Training models...
  [OK] Fuzzy Logic initialized
  [OK] Decision Tree trained

Measuring Fuzzy Logic performance...
  [OK] Inference complete (3421.45 ms total)

Measuring Decision Tree performance...
  [OK] Inference complete (1203.78 ms total)

======================================================================
COMPREHENSIVE COMPARISON: FUZZY LOGIC vs DECISION TREE
======================================================================

EFFECTIVENESS METRICS (Higher is Better)
Metric               Fuzzy Logic         Decision Tree           Winner
Accuracy             0.355               0.5675                  Decision Tree
Precision (macro)    0.3842              0.5803                  Decision Tree
Recall (macro)       0.355               0.5675                  Decision Tree
F1-Score (macro)     0.2939              0.4785                  Decision Tree

COMPUTATIONAL EFFICIENCY (Lower is Better)
Metric                         Fuzzy Logic         Decision Tree          Winner
Avg Time per Sample (ms)       2.8507              1.0032                 Decision Tree
Total Time (ms)                3421.45             1203.78                Decision Tree
Peak Memory Usage (MB)         5.24                2.18                   Decision Tree

======================================================================
[OK] Results saved to: results/comparison.json
```

---

## 6. Input/Output Example (Current Patient)

**Example input (Streamlit sliders):**
```
fever = 7.5
cough = 6.0
sore_throat = 5.0
breath_short = 3.0
fatigue = 6.0
wbc = 8.2
crp = 22.0
spo2 = 95.0
xray_infiltrate = 0
```

**Example output (current patient):**
```
Fuzzy Logic prediction: flu
Decision Tree prediction: flu
Risk probability: 0.58 (Moderate)
Risk score: 0.53
Top Fuzzy rules:
- FLU_R1: fever_high & fatigue_high & sore_high -> 0.62
- FLU_R2: fever_mid & fatigue_high & crp_low -> 0.44
```

---

## 7. Comparison Summary

### A. Computational Efficiency

**Processing Speed**:
- Fuzzy Logic: 2-5 ms per patient
- Decision Tree: 1-2 ms per patient
- **Winner**: Decision Tree (2-3x faster)

**Memory Usage**:
- Fuzzy Logic: ~5 MB
- Decision Tree: ~2 MB
- **Winner**: Decision Tree (60% less)

### B. Effectiveness

**Accuracy**:
- Fuzzy Logic: 35.5%
- Decision Tree: 56.75%
- **Winner**: Decision Tree (+58% improvement)

**F1-Score (Macro)**:
- Fuzzy Logic: 0.2939
- Decision Tree: 0.4785
- **Winner**: Decision Tree (63% better)

**Interpretability**:
- Fuzzy Logic: 100% (explicit rules visible)
- Decision Tree: ~60% (tree structure)
- **Winner**: Fuzzy Logic

### C. Development Experience

**Difficulty**:
- Fuzzy Logic: Very High (5/5)
  - Requires domain expertise
  - Manual rule design
  - Membership function tuning
- Decision Tree: Moderate (3/5)
  - Standard ML pipeline
  - Automated learning
  - Library-based

**Development Time**:
- Fuzzy Logic: ~40 hours (design + testing)
- Decision Tree: ~15 hours (standard workflow)
- **Winner**: Decision Tree (62% faster)

**Code Complexity**:
- Fuzzy Logic: High (custom algorithms)
- Decision Tree: Low (library-based)
- **Winner**: Decision Tree (easier to maintain)

---

## 8. Troubleshooting

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Permission Denied (Windows PowerShell)
```bash
.venv\Scripts\activate.bat
```

### Dataset Missing
```bash
python src/generate_data.py
```

### Streamlit Port Already in Use
```bash
streamlit run src/app.py --server.port 8502
```

---

## 9. Output Files

**Metrics**: `results/metrics.json`
```json
{
  "effectiveness": {
    "fuzzy_logic": {
      "accuracy": 0.355,
      "f1_macro": 0.2939
    },
    "decision_tree": {
      "accuracy": 0.5675,
      "f1_macro": 0.4785
    }
  },
  "cross_validation": {
    "folds": 5,
    "decision_tree": {
      "metrics": {
        "accuracy": {"mean": 0.56, "std": 0.02}
      },
      "per_class_f1": {
        "flu": {"mean": 0.62, "std": 0.04}
      }
    }
  },
  "diagnostics": {
    "classes": ["allergy", "bronchitis", "flu", "pneumonia"],
    "confusion_matrix": {
      "fuzzy_logic": [[...]],
      "decision_tree": [[...]]
    },
    "roc": {
      "fuzzy_logic": {
        "macro_auc": 0.45,
        "per_class": {"flu": {"fpr": [...], "tpr": [...]}}
      }
    }
  },
  "metadata": {
    "dataset": "data/synthetic_patients.csv",
    "label_column": "diagnosis",
    "num_samples": 1200
  }
}
```

**Comparison report**: `results/comparison.json`

---

## 10. Files & Their Purpose

| File | Purpose |
|------|---------|
| `main.py` | Entry point with CLI arguments |
| `src/compare.py` | Comprehensive comparison script |
| `src/evaluate.py` | Model evaluation utilities |
| `src/fuzzy_diagnosis.py` | Fuzzy logic engine (8 rules) |
| `src/ml_decision_tree.py` | Decision tree classifier |
| `requirements.txt` | Python package dependencies |

---

## 11. Code Quality

- Type hints for all functions
- Comprehensive docstrings
- Relative paths (no hard-coding)
- Modular design
- Error handling

## 12. Tests

```bash
python -m unittest
```

---

## 13. Formal Methodology (Exam-Ready)

**Problem**  
Build an interpretable medical decision support system and compare it with a data-driven model.

**Data**  
Synthetic patient dataset with 9 clinical features and 4 diagnosis classes.

**Models**  
- Fuzzy Logic (rule-based, interpretable)
- Decision Tree (supervised machine learning)

**Procedure**  
1. Collect input symptoms and clinical tests.  
2. Run Fuzzy Logic inference to get rule-based scores.  
3. Run Decision Tree inference to get class probabilities.  
4. Compute a risk probability using a weighted clinical score.  
5. Compare predictions, display confidence and diagnostics.  

**Evaluation**  
Compare accuracy, precision, recall, F1, and efficiency metrics across models.

---

## 14. Math Derivations (Simple, Clear)

### 14.1 Fuzzy Membership Functions
Each input is mapped to a membership degree in [0, 1] using triangular or trapezoidal functions.

**Triangular**:
```
mu(x) = 0                    if x <= a or x >= c
mu(x) = (x - a)/(b - a)      if a < x < b
mu(x) = (c - x)/(c - b)      if b < x < c
mu(x) = 1                    if x = b
```

**Trapezoidal**:
```
mu(x) = 0                    if x <= a or x >= d
mu(x) = (x - a)/(b - a)      if a < x < b
mu(x) = 1                    if b <= x <= c
mu(x) = (d - x)/(d - c)      if c < x < d
```

### 14.2 Fuzzy Rule Strength
Rules use min / max logic:
```
AND(a, b, c) = min(a, b, c)
OR(a, b, c)  = max(a, b, c)
```
Each rule produces a disease score; the highest score wins.

### 14.3 Risk Probability
Inputs are normalized to [0,1], then combined:
```
R = sum( weight_i * normalized_i )
P = 1 / (1 + exp(-12 * (R - 0.5)))
```
Risk level is Low / Moderate / High based on thresholds 0.33 and 0.66.

### 14.4 Decision Tree Prediction
The decision tree learns if/else thresholds from data.  
For a new input, it outputs:
```
P(class_k | x)
```
The class with highest probability is chosen.

---

## 15. Extra Evaluation Metrics

The project can report:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Per-class F1
- Confusion matrix
- ROC-AUC (macro and per class)
- Inference time and memory usage

Metrics are saved in `results/metrics.json` and can be visualized in the Streamlit app.

---

## 16. Deployment Notes (Exam Quick Start)

1) Create virtual environment  
2) Install requirements: `pip install -r requirements.txt`  
3) Run tests: `python -m unittest`  
4) Run demo: `python -m src.run_demo`  
5) Launch UI: `streamlit run src/app.py`  

If dataset is missing, regenerate with:
```
python src/generate_data.py
```

---

## 17. Citations / References

- L. A. Zadeh, “Fuzzy Sets,” Information and Control, 1965.  
- J. R. Quinlan, “Induction of Decision Trees,” Machine Learning, 1986.  
- Scikit-learn documentation: DecisionTreeClassifier and model evaluation metrics.  
- Streamlit documentation for UI rendering and layout.  

---

## 18. Limitations (Academic Note)

- Data is synthetic and not clinically validated.  
- Fuzzy rules are simplified for demonstration.  
- Real deployment would require clinical trials and regulatory approval.  

---

## Exam Checklist

See `EXAM_GUIDE.md` for a concise walkthrough, demo steps, and talking points.

---

**Status**: Complete and Functional  
**Last Updated**: January 12, 2026
