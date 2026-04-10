#!/usr/bin/env python3
"""
Presentation Demo: Medical Diagnosis AI System
Shows 4 diverse patient cases with both model predictions side-by-side.
"""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fuzzy_diagnosis import FuzzyDiagnosis, predict_label
from src.ml_decision_tree import FEATURES_CAT, FEATURES_NUM, train_decision_tree


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}".center(80))
    print("="*80)


def print_patient_case(case_num: int, patient_data: dict, true_label: str = None):
    """Print patient case information."""
    print(f"\n[PATIENT] CASE #{case_num}")
    print("-" * 80)
    
    print("\n  SYMPTOMS (0-10 scale):")
    print(f"    - Fever:            {patient_data['fever']:6.1f}")
    print(f"    - Cough:            {patient_data['cough']:6.1f}")
    print(f"    - Sore Throat:      {patient_data['sore_throat']:6.1f}")
    print(f"    - Shortness Breath: {patient_data['breath_short']:6.1f}")
    print(f"    - Fatigue:          {patient_data['fatigue']:6.1f}")
    
    print("\n  LAB TESTS:")
    print(f"    - WBC (x10^9/L):    {patient_data['wbc']:6.1f}")
    print(f"    - CRP (mg/L):       {patient_data['crp']:6.1f}")
    print(f"    - SpO2 (%):         {patient_data['spo2']:6.1f}")
    print(f"    - X-ray Infiltrate: {'YES' if patient_data['xray_infiltrate'] else 'NO':>6}")
    
    if true_label:
        print(f"\n  TRUE DIAGNOSIS: {true_label.upper()}")


def print_predictions(fuzzy_result, fuzzy_pred: str, ml_pred: str, ml_proba: dict):
    """Print model predictions with scores."""
    print("\n" + "-" * 80)
    print("  MODEL PREDICTIONS")
    print("-" * 80)
    
    # Fuzzy Logic Results
    print("\n  [FUZZY] FUZZY LOGIC INFERENCE:")
    print(f"     Diagnosis: {fuzzy_pred.upper()}")
    print("     Confidence Scores:")
    for disease, score in sorted(fuzzy_result.scores.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 20)
        print(f"       - {disease.capitalize():12} {score:.1%}  {bar}")
    
    print("\n     Top Firing Rules:")
    for rule_name, strength in fuzzy_result.fired_rules[:3]:
        print(f"       [OK] {rule_name}: {strength:.2f}")
    
    # Decision Tree Results
    print("\n  [TREE] DECISION TREE CLASSIFICATION:")
    print(f"     Diagnosis: {ml_pred.upper()}")
    print("     Confidence Scores:")
    for disease, prob in sorted(ml_proba.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob * 20)
        print(f"       - {disease.capitalize():12} {prob:.1%}  {bar}")
    
    # Agreement
    print("\n  [MATCH] AGREEMENT:")
    if fuzzy_pred == ml_pred:
        print(f"     [YES] Both models agree: {fuzzy_pred.upper()}")
        print("     (High confidence in diagnosis)")
    else:
        print(f"     [NO] Models differ:")
        print(f"        Fuzzy Logic -> {fuzzy_pred.upper()}")
        print(f"        Decision Tree -> {ml_pred.upper()}")
        print("     (Recommend physician review)")


def main():
    """Run presentation demo."""
    
    print_header("MEDICAL DIAGNOSIS AI - PRESENTATION DEMO")
    print("\nThis demonstration showcases the dual-model approach:")
    print("  1. Fuzzy Logic: Rule-based, fully interpretable diagnosis")
    print("  2. Decision Tree: Data-driven pattern recognition")
    
    # Load data
    print_header("LOADING DATA & TRAINING MODELS")
    df = pd.read_csv(BASE_DIR / "data" / "synthetic_patients.csv")
    print(f"\n[OK] Loaded {len(df)} synthetic patient records")
    
    # Train models
    ml_model = train_decision_tree(df)
    fuzzy_engine = FuzzyDiagnosis()
    print("[OK] Trained Decision Tree (max_depth=5)")
    print("[OK] Initialized Fuzzy Logic Engine (8 rules)")
    
    # Select diverse cases
    print_header("SELECTING DIVERSE PATIENT CASES FOR DEMONSTRATION")
    
    # Case 1: Clear Pneumonia
    idx1 = 5
    # Case 2: Likely Flu
    idx2 = 42
    # Case 3: Allergy
    idx3 = 88
    # Case 4: Random challenging case
    idx4 = 167
    
    cases = [idx1, idx2, idx3, idx4]
    
    for case_num, idx in enumerate(cases, 1):
        row = df.iloc[idx]
        patient_dict = row[FEATURES_NUM + FEATURES_CAT].to_dict()
        true_label = row["diagnosis"]
        
        print_patient_case(case_num, patient_dict, true_label)
        
        # Get predictions
        fuzzy_res = fuzzy_engine.infer(patient_dict)
        fuzzy_pred = predict_label(fuzzy_res.scores)
        
        ml_pred = ml_model.model.predict(pd.DataFrame([row[FEATURES_NUM + FEATURES_CAT]]))[0]
        ml_proba_arr = ml_model.model.predict_proba(pd.DataFrame([row[FEATURES_NUM + FEATURES_CAT]]))[0]
        classes = ml_model.model.named_steps["clf"].classes_
        ml_proba = {c: float(p) for c, p in zip(classes, ml_proba_arr)}
        
        print_predictions(fuzzy_res, fuzzy_pred, ml_pred, ml_proba)
        
        # Accuracy check
        fuzzy_correct = "[OK]" if fuzzy_pred == true_label else "[NO]"
        tree_correct = "[OK]" if ml_pred == true_label else "[NO]"
        print(f"\n  ACCURACY CHECK:")
        print(f"    Fuzzy Logic: {fuzzy_correct} (Predicted: {fuzzy_pred}, Actual: {true_label})")
        print(f"    Decision Tree: {tree_correct} (Predicted: {ml_pred}, Actual: {true_label})")
    
    # Summary statistics
    print_header("MODEL PERFORMANCE SUMMARY (Full Dataset)")
    
    fz_preds = []
    ml_preds = []
    for _, row in df.iterrows():
        patient_dict = row[FEATURES_NUM + FEATURES_CAT].to_dict()
        
        fz_res = fuzzy_engine.infer(patient_dict)
        fz_pred = predict_label(fz_res.scores)
        fz_preds.append(fz_pred)
        
        ml_pred = ml_model.model.predict(pd.DataFrame([row[FEATURES_NUM + FEATURES_CAT]]))[0]
        ml_preds.append(ml_pred)
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    fz_acc = accuracy_score(df["diagnosis"], fz_preds)
    ml_acc = accuracy_score(df["diagnosis"], ml_preds)
    fz_f1 = f1_score(df["diagnosis"], fz_preds, average="macro", zero_division=0)
    ml_f1 = f1_score(df["diagnosis"], ml_preds, average="macro", zero_division=0)
    
    print(f"\n[FUZZY] FUZZY LOGIC PERFORMANCE:")
    print(f"    Accuracy: {fz_acc:.1%}")
    print(f"    F1-Score (Macro): {fz_f1:.4f}")
    
    print(f"\n[TREE] DECISION TREE PERFORMANCE:")
    print(f"    Accuracy: {ml_acc:.1%}")
    print(f"    F1-Score (Macro): {ml_f1:.4f}")
    
    # Advantages summary
    print_header("KEY ADVANTAGES OF DUAL-MODEL APPROACH")
    
    print("\n[FUZZY] FUZZY LOGIC:")
    print("    [OK] Full interpretability - every diagnosis traceable to rules")
    print("    [OK] Medical expert knowledge embedded explicitly")
    print("    [OK] Clear explanation for physician review")
    print("    [OK] Graceful handling of uncertainty")
    
    print("\n[TREE] DECISION TREE:")
    print("    [OK] Data-driven pattern recognition")
    print("    [OK] Learns from training data patterns")
    print("    [OK] Captures complex feature interactions")
    print("    [OK] High accuracy on similar cases")
    
    print("\n[SYSTEM] COMBINED SYSTEM:")
    print("    [OK] Cross-validation through model agreement")
    print("    [OK] Confidence boost when both agree")
    print("    [OK] Flag uncertain cases for physician review")
    print("    [OK] Educational value - understand both paradigms")
    
    print_header("RUNNING THE SYSTEM")
    
    print("\n[WEB] WEB INTERFACE (Interactive Demo):")
    print("    Command: streamlit run src/app.py")
    print("    Features:")
    print("      - Real-time patient data entry via sliders")
    print("      - Live predictions from both models")
    print("      - Visual comparison charts")
    print("      - Model agreement indicators")
    print("      - Explainable predictions")
    
    print("\n[CLI] COMMAND LINE (Batch Processing):")
    print("    Command: python src/run_demo.py")
    print("    Output: Single patient example with detailed predictions")
    
    print("\n[EVAL] EVALUATION (Metrics Computation):")
    print("    Command: python src/evaluate.py")
    print("    Output: Full dataset metrics in results/metrics.json")
    
    print_header("END OF DEMONSTRATION")
    print("\n[OK] Demo complete. All systems operational and ready for presentation.\n")


if __name__ == "__main__":
    main()
