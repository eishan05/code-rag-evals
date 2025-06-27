import os
import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict

# --- Dependency Imports ---
import coir
import numpy as np
import matplotlib
# Use a non-interactive backend suitable for containers without a GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import HFModel

try:
    from google import generativeai as genai
except ImportError:
    print("Google Generative AI library not found. It should be installed via requirements.txt.")
    genai = None

# --- Configuration ---
# TODO(eishanlawrence): Make this configurable through flags.
MODELS_TO_EVALUATE = [
    "gemini",
    "sentence-transformers/all-MiniLM-L6-v2",
    # "Salesforce/SFR-Embedding-Code-400M_R",
]
TASKS_TO_EVALUATE = ["apps"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Model Definitions ---
class GeminiAPIModel:
    """
    A wrapper for the Google Gemini API (text-embedding-004) to be used with COIR.
    Reads the API key from the GOOGLE_API_KEY environment variable.
    """
    def __init__(self, **kwargs):
        if not genai:
            raise ImportError("Google Generative AI SDK is required for GeminiAPIModel.")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your API key.")
        
        genai.configure(api_key=api_key)
        self.model_name = "models/text-embedding-004"
        self.requests_per_minute = 300  # Max requests per minute for text-embedding-004
        self.delay_between_requests = 60 / self.requests_per_minute

    def encode_text(self, texts: list, batch_size: int, task_type: str) -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts with task_type='{task_type}'...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding ({task_type})", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            try:
                result = genai.embed_content(
                    model=self.model_name, content=batch_texts, task_type=task_type
                )
                all_embeddings.extend(result['embedding'])
            except Exception as e:
                logging.error(f"An API error occurred on a batch: {e}")
                # Add zero vectors for the failed batch to avoid downstream errors
                all_embeddings.extend([np.zeros(768)] * len(batch_texts)) # Assumes 768 dimensions for gemini-004
            
            # Simple rate limiting
            time.sleep(self.delay_between_requests)
            
        return np.array(all_embeddings, dtype=np.float32)

    def encode_queries(self, queries: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        return self.encode_text(queries, batch_size=batch_size, task_type="RETRIEVAL_QUERY")

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 100, **kwargs) -> np.ndarray:
        texts = [doc['text'] for doc in corpus]
        return self.encode_text(texts, batch_size=batch_size, task_type="RETRIEVAL_DOCUMENT")


# --- Evaluation Function ---
def run_evaluations(model_list: List[str], task_list: List[str]):
    """
    Runs the COIR evaluation for each model and task combination.
    """
    print("\n--- Starting Model Evaluations ---")
    for model_name in model_list:
        safe_model_name = model_name.replace('/', '_')
        print(f"\n{'='*40}\nEvaluating model: {model_name}\n{'='*40}")
        try:
            if model_name.lower() == "gemini":
                model = GeminiAPIModel()
            else:
                model = HFModel(model_name=model_name)
        except (ValueError, ImportError) as e:
            logging.error(f"Could not load model '{model_name}'. Skipping. Reason: {e}")
            continue

        for task_name in task_list:
            print(f"\n--- Running task: {task_name} ---")
            tasks = get_tasks(tasks=[task_name])
            evaluation = COIR(tasks=tasks, batch_size=100)
            
            output_folder = f"results/{safe_model_name}"
            # The Dockerfile creates the base 'results' directory
            os.makedirs(output_folder, exist_ok=True)
            
            results = evaluation.run(model, output_folder=output_folder)
            print(f"--- Finished task: {task_name} ---\nResults: {results}")

    print("\n✅ All evaluations complete!")


# --- Visualization Functions (MODIFIED TO SAVE PLOTS) ---
def load_all_results_for_task(task_name: str, model_list: List[str]) -> Dict:
    """Loads all result files for a given task from the results directory."""
    all_results = {}
    for model_name in model_list:
        safe_model_name = model_name.replace('/', '_')
        file_path = Path(f"results/{safe_model_name}/{task_name}.json")
        if file_path.exists():
            with open(file_path, 'r') as f:
                all_results[model_name] = json.load(f).get("metrics", {})
        else:
            print(f"Warning: Result file not found for model '{model_name}'. Skipping from plot.")
    return all_results

def plot_full_comparison(task_name: str, model_list: List[str]):
    """Generates and saves a plot comparing models across all K values for a task."""
    comparison_data = load_all_results_for_task(task_name, model_list)
    if not comparison_data:
        print(f"No data to plot for task: {task_name}")
        return

    metric_names = list(next(iter(comparison_data.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle(f'Model Comparison on "{task_name}" Task', fontsize=20)

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        k_vals_for_ticks = set()
        for model_name, metrics in comparison_data.items():
            if metric_name in metrics:
                values_dict = metrics[metric_name]
                k_values, scores = [], []
                for key, score in values_dict.items():
                    match = re.search(r'\d+$', key)
                    if match:
                        k = int(match.group())
                        k_values.append(k)
                        scores.append(score)
                        k_vals_for_ticks.add(k)
                if k_values:
                    sorted_pairs = sorted(zip(k_values, scores))
                    k_vals, score_vals = zip(*sorted_pairs)
                    ax.plot(k_vals, score_vals, marker='o', linestyle='-', label=model_name)
        
        ax.set_title(f'{metric_name} vs. K', fontsize=16)
        ax.set_xlabel('K (cutoff)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.legend(fontsize=10)
        
        if k_vals_for_ticks:
            all_k = sorted(list(k_vals_for_ticks))
            ax.set_xscale('log')
            ax.set_xticks(all_k)
            ax.set_xticklabels(all_k, rotation=45)
    
    output_path = f"output_plots/{task_name}_full_comparison.png"
    plt.savefig(output_path)
    plt.close(fig) # Free up memory
    print(f"Saved full comparison plot to {output_path}")

def plot_k1_comparison(task_name: str, model_list: List[str]):
    """Generates and saves a bar chart comparing all models at K=1."""
    comparison_data = load_all_results_for_task(task_name, model_list)
    if not comparison_data:
        print(f"No K=1 data to plot for task: {task_name}")
        return

    metrics_at_1 = ['NDCG@1', 'MAP@1', 'Recall@1', 'P@1']
    valid_models = [m for m in model_list if m in comparison_data]
    plot_data = {model: [] for model in valid_models}

    for metric_key in metrics_at_1:
        main_metric = metric_key.split('@')[0]
        if main_metric == "P": main_metric = "Precision"
        for model_name in valid_models:
            plot_data[model_name].append(comparison_data[model_name].get(main_metric, {}).get(metric_key, 0))
    
    if not any(plot_data.values()):
        print(f"No valid K=1 data found for any model on task {task_name}.")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    num_models = len(valid_models)
    bar_width = 0.2
    index = np.arange(len(metrics_at_1))
    
    for i, (model_name, scores) in enumerate(plot_data.items()):
        pos = index - ((num_models - 1) * bar_width / 2) + (i * bar_width)
        bars = ax.bar(pos, scores, bar_width, label=model_name)
        ax.bar_label(bars, padding=3, fmt='%.3f', fontsize=10)
        
    ax.set_title(f'Model Comparison at K=1 for "{task_name}" Task', fontsize=18, pad=20)
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_at_1, fontsize=12)
    ax.legend(title='Models', bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.yaxis.grid(True, linestyle='--', alpha=.25)
    max_score = max(max(s) for s in plot_data.values() if s) if any(any(s) for s in plot_data.values()) else 1
    ax.set_ylim(bottom=0, top=max_score * 1.15)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path = f"output_plots/{task_name}_k1_comparison.png"
    plt.savefig(output_path)
    plt.close(fig) # Free up memory
    print(f"Saved K=1 comparison plot to {output_path}")

def run_visualizations(model_list: List[str], task_list: List[str]):
    """Generates and saves plots for each task's results."""
    print("\n--- Generating Visualizations ---")
    # The Dockerfile creates the base 'output_plots' directory
    # os.makedirs("output_plots", exist_ok=True)
    for task_name in task_list:
        print(f"\n# Visualizing results for: {task_name}")
        plot_full_comparison(task_name, model_list)
        plot_k1_comparison(task_name, model_list)
    print("\n✅ All visualizations generated.")

if __name__ == "__main__":
    # Step 1: Run evaluations and save results to the 'results/' directory
    run_evaluations(MODELS_TO_EVALUATE, TASKS_TO_EVALUATE)

    # Step 2: Generate plots from the results and save them to 'output_plots/'
    run_visualizations(MODELS_TO_EVALUATE, TASKS_TO_EVALUATE)

    print("\n✅ Script finished successfully.")