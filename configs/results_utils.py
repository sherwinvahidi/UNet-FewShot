# configs/results_utils.py
import json
import os


def load_results(results_dir, filename):
    """Load JSON results file. Returns None if not found."""
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    print(f"⚠ {filename} not found")
    return None


def save_kshot_results(results, results_dir, filename):
    """Save k-shot results dict to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    serializable = {str(k): {'mean': float(v['mean']), 'std': float(v['std'])}
                    for k, v in results.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=4)
    print(f"✓ Saved to: {path}")


def print_kshot_results(results, k_values, label="K-SHOT RESULTS"):
    """Print formatted k-shot results table."""
    print(f"\n{'='*50}")
    print(label)
    print(f"{'='*50}")
    for k in k_values:
        key = str(k) if str(k) in results else k
        r = results[key]
        print(f"  k={k:>2}: Dice = {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"{'='*50}")


def print_comparison_table(all_results, k_values):
    """
    Print multi-method comparison table.

    Args:
        all_results: list of (name, results_dict) tuples
        k_values: list of k values
    """
    header = f"{'Method':<25}"
    for k in k_values:
        header += f" {'k='+str(k):<12}"
    print(f"\n{'='*70}")
    print("METHOD COMPARISON")
    print(f"{'='*70}")
    print(header)
    print(f"{'-'*70}")
    for name, results in all_results:
        if results is None:
            continue
        row = f"{name:<25}"
        for k in k_values:
            r = results[str(k)]
            row += f" {r['mean']:.3f}±{r['std']:.3f}  "
        print(row)
    print(f"{'='*70}")