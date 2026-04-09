import json
import os
import glob
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Plots will not be generated. Use pip install matplotlib")
    HAS_MATPLOTLIB = False

def aggregate_results(results_dir="results"):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No json files found in {results_dir}")
        return

    # Structure: aggregated[model_name][metric_name] = [values]
    aggregated = defaultdict(lambda: defaultdict(list))
    # Structure for curves: raw_data[model_name][method_name] = {'scores': list, 'labels': list} (from the best run or concatenated)
    
    # We will compute the Mean and Std across runs.
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            
        mod = data["model_name"]
        
        aggregated[mod]["classification_accuracy"].append(data["classification_accuracy"])
        aggregated[mod]["siamese_accuracy"].append(data["siamese_accuracy"])
        aggregated[mod]["localization_accuracy"].append(data["localization_accuracy"])
        
        for method, metrics in data["ood_metrics"].items():
            aggregated[mod][f"{method}_AUROC"].append(metrics["AUROC"])
            aggregated[mod][f"{method}_AUPR"].append(metrics["AUPR"])
            aggregated[mod][f"{method}_FPR95"].append(metrics["FPR95"])

    # Output Markdown Table
    md_lines = ["# Statistical Results across Runs\n"]
    for mod, metrics in aggregated.items():
        md_lines.append(f"## Model: {mod}\n")
        md_lines.append("| Metric | Mean | Std Dev | Best |")
        md_lines.append("|---|---|---|---|")
        
        for metric_name, values in sorted(metrics.items()):
            mean_val = np.mean(values)
            std_val = np.std(values)
            best_val = np.max(values) if "FPR95" not in metric_name else np.min(values)
            
            md_lines.append(f"| {metric_name} | {mean_val:.4f} | {std_val:.4f} | {best_val:.4f} |")
        md_lines.append("\n")

    with open(os.path.join(results_dir, "aggregated_table.md"), "w") as f:
        f.write("\n".join(md_lines))
    print("Generated aggregated_table.md")

    if not HAS_MATPLOTLIB:
        return

    # Generate Bar Chart for OOD AUROC
    for mod, metrics in aggregated.items():
        methods = []
        means = []
        stds = []
        for metric_name, values in metrics.items():
            if metric_name.endswith("_AUROC"):
                methods.append(metric_name.replace("_AUROC", ""))
                means.append(np.mean(values))
                stds.append(np.std(values))
                
        if methods:
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(methods))
            plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
            plt.xticks(x_pos, methods, rotation=45, ha='right')
            plt.ylabel('AUROC')
            plt.title(f'OOD Detection AUROC ({mod}) - Mean ± Std')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{mod}_auroc_barchart.png"))
            plt.close()

    # Generate ROC Curve for the FIRST run found
    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        
        mod = data["model_name"]
        plt.figure(figsize=(8, 8))
        
        for method, metrics in data["ood_metrics"].items():
            if "labels" in metrics and "scores" in metrics:
                y_true = metrics["labels"]
                y_score = metrics["scores"]
                
                if len(np.unique(y_true)) > 1:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    plt.plot(fpr, tpr, label=f'{method} (AUC = {metrics["AUROC"]:.3f})')
                    
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({mod})')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f"{mod}_roc_curve.png"))
        plt.close()
        break # Only plot ROC for one representative run to avoid overlap confusion

if __name__ == "__main__":
    aggregate_results()
