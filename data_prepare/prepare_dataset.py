#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preparation Script
Extract features from multiple projects and generate training/test datasets
"""

import os
import sys
import subprocess
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
                        # Normalize path format (use forward slashes) and normalize path
                        file_path = file_path.replace('\\', '/')
                        # Remove leading ./ or /
                        if file_path.startswith('./'):
                            file_path = file_path[2:]
                        elif file_path.startswith('/'):
                            file_path = file_path[1:]
                        
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
                # Normalize path format (use forward slashes) and normalize path
                file_path = line.replace('\\', '/')
                # Remove leading ./ or /
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                elif file_path.startswith('/'):
                    file_path = file_path[1:]
                commit_count[file_path] += 1
                if current_author:
                    author_count[file_path].add(current_author)
        
        # Method 2: If churn is 0, try using git log --shortstat as fallback
        if sum(churn.values()) == 0:
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
        print(f"  Warning: Git command execution failed: {e}")
    
    # Convert author_count to count
    author_count_dict = {k: len(v) for k, v in author_count.items()}
    
    return dict(commit_count), dict(churn), author_count_dict


def estimate_complexity_simple(file_path):
    """
    Simple complexity estimation (based on control flow keywords)
    Used when radon is not available
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return 0
    
    # Python control flow keywords
    keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'and', 'or', 'not']
    complexity = 1  # Base complexity
    
    for keyword in keywords:
        # Use word boundary matching to avoid false matches
        pattern = r'\b' + re.escape(keyword) + r'\b'
        matches = len(re.findall(pattern, content, re.IGNORECASE))
        complexity += matches
    
    return complexity


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
            # Normalize path format (use forward slashes)
            rel_path = str(py_file.relative_to(repo_path)).replace('\\', '/')
            
            # Calculate LOC first (needed for MI calculation later)
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    loc = sum(1 for line in f if line.strip())
            except Exception:
                loc = 0
            
            # Use radon to get complexity
            try:
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
                avg_cc = estimate_complexity_simple(py_file)
            
            # Use radon to get MI score
            mi_score = None
            try:
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
                # Normalize path format (use forward slashes) and normalize path
                file_path = line.replace('\\', '/')
                # Remove leading ./ or /
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                elif file_path.startswith('/'):
                    file_path = file_path[1:]
                labels[file_path] = 1
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Warning: Error generating labels: {e}")
    
    return dict(labels)


def extract_features_from_repo(repo_path, project_name=None):
    """
    Extract all features from a single repository
    Returns: DataFrame containing features for all files
    """
    repo_path = Path(repo_path).resolve()
    
    if project_name is None:
        project_name = repo_path.name
    
    print(f"\nProcessing project: {project_name}")
    print(f"Path: {repo_path}")
    
    # Check if it's a Git repository
    if not (repo_path / ".git").exists():
        print(f"  Warning: {repo_path} is not a Git repository, skipping")
        return pd.DataFrame()
    
    # Extract Git metrics
    print("  Extracting Git metrics...")
    commit_count, churn, author_count = get_git_metrics(repo_path)
    
    # Extract code quality metrics
    print("  Analyzing code quality...")
    quality = analyze_code_quality(repo_path)
    
    # Generate labels
    print("  Generating labels...")
    bug_labels = build_labels(repo_path)
    
    # Merge all files
    print("  Building dataset...")
    # Prefer files in quality (contains all Python files), then supplement with git files
    all_files = set(quality.keys())
    all_files.update(commit_count.keys())
    
    rows = []
    for file in all_files:
        # Only process Python files
        if not file.endswith('.py'):
            continue
        
        q = quality.get(file, {})
        rows.append({
            "project": project_name,
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
    
    # If files exist in quality but not in git, at least keep files with loc
    # Filter out files with no data at all
    if len(df) > 0:
        # At least need commit or loc
        df = df[(df["commit_count"] > 0) | (df["loc"] > 0)]
    
    print(f"  Complete: Extracted {len(df)} files")
    if len(df) > 0:
        print(f"  - Files with commits: {(df['commit_count'] > 0).sum()}")
        print(f"  - Files with code quality data: {(df['loc'] > 0).sum()}")
    
    return df


def prepare_dataset(repo_paths, output_file="dataset.csv"):
    """
    Extract features from multiple projects and generate dataset
    
    Parameters:
        repo_paths: List of project paths
        output_file: Output CSV filename
    """
    print("=" * 60)
    print("Data Preparation: Extracting Features from Multiple Projects")
    print("=" * 60)
    print(f"\nNumber of projects: {len(repo_paths)}")
    
    all_dataframes = []
    
    for i, repo_path in enumerate(repo_paths, 1):
        repo_path = Path(repo_path).resolve()
        
        if not repo_path.exists():
            print(f"\n[{i}/{len(repo_paths)}] Skip: Path does not exist - {repo_path}")
            continue
        
        project_name = repo_path.name
        
        try:
            df = extract_features_from_repo(repo_path, project_name)
            if len(df) > 0:
                all_dataframes.append(df)
                print(f"  [OK] Successfully extracted {len(df)} files")
            else:
                print(f"  [WARN] No data extracted")
        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Merge all data
    if not all_dataframes:
        print("\nError: No data successfully extracted")
        return
    
    print("\n" + "=" * 60)
    print("Merging dataset...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Reorder columns
    columns_order = ["project", "file", "commit_count", "churn", "author_count", "cc", "mi", "loc", "label"]
    final_df = final_df[columns_order]
    
    # Save to CSV (None values will be saved as empty, pandas handles automatically)
    output_path = Path(output_file)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig", na_rep='')
    
    print(f"\nDataset Statistics:")
    print(f"  Total files: {len(final_df)}")
    print(f"  Number of projects: {final_df['project'].nunique()}")
    print(f"  Feature columns: {', '.join(columns_order[2:])}")
    
    print(f"\nFiles per project:")
    project_counts = final_df['project'].value_counts()
    for project, count in project_counts.items():
        print(f"  {project}: {count} files")
    
    print(f"\nFeature Statistics:")
    print(final_df[columns_order[2:]].describe())
    
    print(f"\n[SUCCESS] Dataset saved to: {output_path.absolute()}")
    
    return final_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from multiple Git projects and generate training/test dataset")
    parser.add_argument("repos", nargs="*", help="List of project paths (can be multiple paths)")
    parser.add_argument("-o", "--output", default="dataset.csv", help="Output CSV filename (default: dataset.csv)")
    parser.add_argument("-f", "--file", help="Read project path list from file (one path per line)")
    
    args = parser.parse_args()
    
    # Get project path list
    repo_paths = []
    
    if args.file:
        # Read from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                repo_paths = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        except Exception as e:
            print(f"Error: Cannot read file {args.file}: {e}")
            return
    else:
        # Read from command line arguments
        repo_paths = args.repos
    
    if not repo_paths:
        print("Error: No project paths provided")
        parser.print_help()
        return
    
    # Execute data preparation
    prepare_dataset(repo_paths, args.output)


if __name__ == "__main__":
    main()

