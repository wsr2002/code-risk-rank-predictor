#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning-based Code Risk Predictor
Uses machine learning models to predict file risk/test priority
"""

import os
import sys
import subprocess
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set output encoding to UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Bug fix keywords
BUG_KEYWORDS = ["fix", "bug", "error", "patch", "hotfix", "resolve", "issue", "defect"]


def is_bug_fix(message):
    """Check if commit message contains bug fix keywords"""
    if not message:
        return False
    return any(k in message.lower() for k in BUG_KEYWORDS)


def get_git_metrics(repo_path):
    """
    Extract file-level metrics from git history
    Returns: commit_count, churn, author_count
    """
    repo_path = Path(repo_path).resolve()
    
    commit_count = defaultdict(int)
    churn = defaultdict(int)
    author_count = defaultdict(set)
    
    try:
        # Method 1: Use git log --numstat to get churn data (don't use --name-only as it overrides numstat)
        cmd = ["git", "-C", str(repo_path), "log", "--pretty=format:%an", "--numstat"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=120).decode("utf-8", errors="ignore")
        
        current_author = None
        in_numstat = False
        
        for line in output.splitlines():
            line = line.strip()
            if not line:
                in_numstat = False
                continue
            
            # Check if it's an author line (usually doesn't contain tabs or start with digits)
            if "\t" not in line and not line[0].isdigit():
                # Possibly an author name
                if len(line) < 100 and "/" not in line and "\\" not in line:
                    current_author = line
                    in_numstat = True
                continue
            
            # Check if it's a numstat line (format: added_lines\tdeleted_lines\tfilename)
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    added_str = parts[0].strip()
                    deleted_str = parts[1].strip()
                    file_path = parts[2].strip()
                    
                    # Only process Python files
                    if file_path.endswith('.py'):
                        added = int(added_str) if added_str.isdigit() else 0
                        deleted = int(deleted_str) if deleted_str.isdigit() else 0
                        
                        commit_count[file_path] += 1
                        churn[file_path] += (added + deleted)
                        if current_author:
                            author_count[file_path].add(current_author)
                except (ValueError, IndexError):
                    continue
            elif line.endswith('.py') and in_numstat:
                # If only filename (some git versions may not have numstat)
                commit_count[line] += 1
                if current_author:
                    author_count[line].add(current_author)
        
        # Method 2: If churn is 0, try using git log --shortstat as fallback
        if sum(churn.values()) == 0:
            print("  Using fallback method to calculate churn...")
            for file_path in list(commit_count.keys()):
                try:
                    # Get statistics for all commits of this file
                    cmd = ["git", "-C", str(repo_path), "log", "--shortstat", "--oneline", "--", file_path]
                    stat_output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30).decode("utf-8", errors="ignore")
                    
                    # Parse shortstat output
                    for line in stat_output.splitlines():
                        if "file changed" in line or "files changed" in line:
                            # Extract numbers
                            numbers = re.findall(r'(\d+)', line)
                            if len(numbers) >= 2:
                                added = int(numbers[0]) if len(numbers) > 0 else 0
                                deleted = int(numbers[1]) if len(numbers) > 1 else 0
                                churn[file_path] += (added + deleted)
                except Exception:
                    continue
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Git command execution failed: {e}")
    
    # Convert author_count to count
    author_count_dict = {k: len(v) for k, v in author_count.items()}
    
    return dict(commit_count), dict(churn), author_count_dict


def analyze_code_quality(repo_path):
    """
    Analyze code quality using radon
    Returns: {file: {"cc": cyclomatic_complexity, "mi": maintainability_index, "loc": lines_of_code}}
    """
    repo_path = Path(repo_path).resolve()
    quality = {}
    
    # Find all Python files
    python_files = list(repo_path.rglob("*.py"))
    
    for py_file in python_files:
        try:
            rel_path = str(py_file.relative_to(repo_path))
            
            # Calculate LOC first (needed for MI calculation later)
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    loc = sum(1 for line in f if line.strip())
            except Exception:
                loc = 0
            
            # Use radon to get complexity
            try:
                import sys
                # Try using python -m radon to invoke
                cc_output = subprocess.check_output(
                    [sys.executable, "-m", "radon", "cc", "-a", "-s", str(py_file)],
                    stderr=subprocess.DEVNULL,
                    timeout=10
                ).decode("utf-8", errors="ignore")
                
                # Parse average complexity
                cc_values = []
                for line in cc_output.splitlines():
                    if "Average complexity:" in line:
                        match = re.search(r'Average complexity:\s*([\d.]+)', line)
                        if match:
                            cc_values.append(float(match.group(1)))
                    # Also try to parse individual function complexity
                    elif re.match(r'^\s+\w+.*\s+\([A-F]\)\s+(\d+)', line):
                        match = re.search(r'\([A-F]\)\s+(\d+)', line)
                        if match:
                            cc_values.append(float(match.group(1)))
                
                avg_cc = np.mean(cc_values) if cc_values else 0
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # If radon is not available, use simple complexity estimation
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'and', 'or', 'not']
                    complexity = 1
                    for keyword in keywords:
                        pattern = r'\b' + re.escape(keyword) + r'\b'
                        matches = len(re.findall(pattern, content, re.IGNORECASE))
                        complexity += matches
                    avg_cc = complexity
                except Exception:
                    avg_cc = 0
            
            # Use radon to get MI score
            mi_score = None
            try:
                import sys
                mi_output = subprocess.check_output(
                    [sys.executable, "-m", "radon", "mi", "-s", str(py_file)],
                    stderr=subprocess.DEVNULL,
                    timeout=10
                ).decode("utf-8", errors="ignore")
                
                # Parse MI score
                mi_match = re.search(r'([\d.]+)\s*\([A-F]\)', mi_output)
                if mi_match:
                    mi_score = float(mi_match.group(1))
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, AttributeError):
                pass  # radon not available, use estimation method
            
            # If radon didn't successfully extract MI, use estimation based on LOC and complexity
            if mi_score is None:
                if loc > 0:
                    # Has code lines, use formula to estimate
                    # MI formula source: Microsoft's Maintainability Index standard formula
                    # Full formula: MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * Halstead Volume / π))
                    # Simplified version (ignoring Halstead Volume and sin term): MI ≈ 171 - 0.23 * CC - 16.2 * ln(LOC)
                    # Reference: https://en.wikipedia.org/wiki/Maintainability_index
                    if avg_cc > 0:
                        mi_score = max(0, min(100, 171 - 0.23 * avg_cc - 16.2 * np.log(loc + 1)))
                    else:
                        # Only LOC, no CC, estimate based on LOC (assuming average complexity)
                        # Assume average complexity is about 10 (medium complexity)
                        estimated_cc = 10
                        mi_score = max(0, min(100, 171 - 0.23 * estimated_cc - 16.2 * np.log(loc + 1)))
                else:
                    # No code (loc=0) and no CC, set to None to indicate missing data
                    mi_score = None
            
            quality[rel_path] = {
                "cc": avg_cc,
                "mi": mi_score,
                "loc": loc
            }
        
        except Exception as e:
            continue
    
    return quality


def build_labels(repo_path):
    """
    Generate labels by parsing commit messages
    Returns: {file: 1 if appears in bug fix commit, else 0}
    """
    repo_path = Path(repo_path).resolve()
    labels = defaultdict(int)
    
    try:
        # Get commit log, including messages and modified files
        cmd = ["git", "-C", str(repo_path), "log", "--name-only", "--pretty=format:%s"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=60).decode("utf-8", errors="ignore")
        
        current_msg = None
        
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # If line doesn't contain a dot (might be commit message)
            if "." not in line or not line.endswith('.py'):
                # Check if it's a commit message (usually doesn't contain path separators)
                if "/" not in line and "\\" not in line:
                    current_msg = line
                continue
            
            # If it's a Python file and current commit is a bug fix
            if line.endswith('.py') and current_msg and is_bug_fix(current_msg):
                labels[line] = 1
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Error generating labels: {e}")
    
    return dict(labels)


def build_training_dataset(repo_path):
    """
    Build training dataset
    Returns: DataFrame containing all features and labels
    """
    print("Extracting Git metrics...")
    commit_count, churn, author_count = get_git_metrics(repo_path)
    
    print("Analyzing code quality...")
    quality = analyze_code_quality(repo_path)
    
    print("Generating labels...")
    bug_labels = build_labels(repo_path)
    
    print("Building dataset...")
    rows = []
    
    # Merge all files
    all_files = set(commit_count.keys()) | set(quality.keys())
    
    for file in all_files:
        # Only process Python files
        if not file.endswith('.py'):
            continue
        
        q = quality.get(file, {})
        rows.append({
            "file": file,
            "commit_count": commit_count.get(file, 0),
            "churn": churn.get(file, 0),
            "author_count": author_count.get(file, 0),
            "cc": q.get("cc", 0),
            "mi": q.get("mi", None),  # If no quality data, set to None (indicates missing data)
            "loc": q.get("loc", 0),
            "label": bug_labels.get(file, 0),
        })
    
    df = pd.DataFrame(rows)
    
    # Filter out files without sufficient data
    df = df[df["commit_count"] > 0]  # At least one commit required
    
    return df


def train_model(df):
    """
    Train machine learning model
    Returns: Trained model
    """
    if len(df) == 0:
        raise ValueError("Dataset is empty, cannot train model")
    
    # Select features
    feature_cols = ["commit_count", "churn", "author_count", "cc", "mi", "loc"]
    X = df[feature_cols].fillna(0)
    y = df["label"]
    
    # Check if there are positive samples
    if y.sum() == 0:
        print("Warning: No positive samples (bug fix), using all files as training data")
        # If no bug fixes, we can use other strategies, such as using high churn as proxy
        y = (df["churn"] > df["churn"].quantile(0.75)).astype(int)
    
    print(f"\nDataset Statistics:")
    print(f"  Total files: {len(df)}")
    print(f"  Positive samples (bug fix): {y.sum()}")
    print(f"  Negative samples: {(y == 0).sum()}")
    
    # Split training and test sets (8:2)
    if len(df) < 10:
        # Too little data, use all as training set
        X_train, X_test = X, X
        y_train, y_test = y, y
        print(f"\nData Split:")
        print(f"  Training set: {len(X_train)} files")
        print(f"  Test set: {len(X_test)} files")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 else None
        )
        print(f"\nData Split (8:2):")
        print(f"  Training set: {len(X_train)} files ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} files ({len(X_test)/len(df)*100:.1f}%)")
        print(f"  Training set positive samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Test set positive samples: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    if len(X_test) > 0:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\nModel Evaluation Results:")
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  AUC-ROC: {auc:.3f}")
        except ValueError:
            print("  AUC-ROC: Cannot calculate (possibly only one class)")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix heatmap (font size increased to 1.5x)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Bug-Fix-Related File', 'Bug-Fix-Related File'],
                    yticklabels=['Non-Bug-Fix-Related File', 'Bug-Fix-Related File'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 30, 'weight': 'bold'})  # 30 = 20 * 1.5
        plt.title('Confusion Matrix Heatmap', fontsize=33, fontweight='bold', pad=30)  # 33 = 22 * 1.5
        plt.ylabel('True Label', fontsize=27)  # 27 = 18 * 1.5
        plt.xlabel('Predicted Label', fontsize=27)  # 27 = 18 * 1.5
        # Increase axis tick label font size
        plt.tick_params(labelsize=24)  # 24 = 16 * 1.5
        # Increase colorbar label font size
        cbar = plt.gca().collections[0].colorbar
        cbar.set_label('Count', fontsize=24)  # 24 = 16 * 1.5
        cbar.ax.tick_params(labelsize=21)  # 21 = 14 * 1.5
        plt.tight_layout()
        
        # Save image
        confusion_matrix_file = 'confusion_matrix.png'
        plt.savefig(confusion_matrix_file, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix heatmap saved to: {confusion_matrix_file}")
        plt.close()
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        # Plot ROC curve (font size increased to 1.5x)
        plt.figure(figsize=(12, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=4, label=f'ROC curve (AUC = {auc:.3f})')  # lw=4 = 3*1.33
        plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--', label='Random Classifier (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=27)  # 27 = 18 * 1.5
        plt.ylabel('True Positive Rate', fontsize=27)  # 27 = 18 * 1.5
        plt.title('ROC Curve', fontsize=33, fontweight='bold', pad=30)  # 33 = 22 * 1.5
        plt.legend(loc="lower right", fontsize=24)  # 24 = 16 * 1.5
        plt.grid(True, alpha=0.3)
        # Increase axis tick label font size
        plt.tick_params(labelsize=24)  # 24 = 16 * 1.5
        plt.tight_layout()
        
        # Save ROC curve plot
        roc_curve_file = 'roc_curve.png'
        plt.savefig(roc_curve_file, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {roc_curve_file}")
        plt.close()
        
        # Print confusion matrix values
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"             Non-Bug   Bug")
        print(f"True  Non-Bug   {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"      Bug       {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Feature importance
        print("\nFeature Importance:")
        feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        print(feature_importance.to_string(index=False))
    
    return model


def predict_risk(model, df):
    """
    Use model to predict file risk scores
    Returns: DataFrame sorted by risk score
    """
    feature_cols = ["commit_count", "churn", "author_count", "cc", "mi", "loc"]
    X = df[feature_cols].fillna(0)
    
    # Predict risk probability
    risk_scores = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["risk"] = risk_scores
    
    # Sort by risk score
    df_sorted = df.sort_values("risk", ascending=False)
    
    return df_sorted


def load_dataset_from_csv(csv_path):
    """
    Load dataset from CSV file
    Returns: DataFrame
    """
    print(f"Loading data from CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ["project", "file", "commit_count", "churn", "author_count", "cc", "mi", "loc", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    print(f"Successfully loaded {len(df)} records")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    return df


def main():
    """Main function"""
    import sys
    
    # Default CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r"data_prepare\python_projects_dataset_final.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file does not exist: {csv_path}")
        print("\nUsage:")
        print("  python hotspot_detection.py [CSV file path]")
        print("\nExample:")
        print("  python hotspot_detection.py data_prepare/python_projects_dataset_final.csv")
        return
    
    print("=" * 60)
    print("Machine Learning-based Hotspot Detection System")
    print("=" * 60)
    print(f"\nDataset file: {csv_path}\n")
    
    try:
        # Load dataset from CSV file
        df = load_dataset_from_csv(csv_path)
        
        if len(df) == 0:
            print("Error: CSV file is empty")
            return
        
        print(f"\nDataset Statistics:")
        print(f"  Total files: {len(df)}")
        print(f"  Number of projects: {df['project'].nunique()}")
        print(f"  Positive samples (label=1): {df['label'].sum()}")
        print(f"  Negative samples (label=0): {(df['label'] == 0).sum()}")
        
        # Train model (internal 8:2 split)
        model = train_model(df)
        
        # Predict on entire dataset (for display purposes)
        print("\n" + "=" * 60)
        print("Top High-Risk Files (Predicted by ML Model)")
        print("=" * 60)
        
        df_risky = predict_risk(model, df)
        
        # Display Top 20 high-risk files
        top_n = min(20, len(df_risky))
        print(f"\nTop {top_n} High-Risk Files:\n")
        print(f"{'Rank':<6} {'Risk Score':<12} {'Project':<15} {'File Path':<50} {'Commits':<10} {'Churn':<10}")
        print("-" * 100)
        
        for idx, row in df_risky.head(top_n).iterrows():
            rank = df_risky.index.get_loc(idx) + 1
            project = row.get('project', 'N/A')
            file_path = row['file'][:48] if len(row['file']) > 48 else row['file']
            print(f"{rank:<6} {row['risk']:<12.3f} {project:<15} {file_path:<50} {row['commit_count']:<10} {row['churn']:<10}")
        
        # Save results to CSV
        output_file = "hotspot_predictions.csv"
        output_cols = ["project", "file", "risk", "commit_count", "churn", "author_count", "cc", "mi", "loc", "label"]
        df_risky[output_cols].to_csv(
            output_file, index=False, encoding="utf-8-sig", na_rep=''
        )
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

