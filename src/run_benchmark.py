import coir
import os
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
from coir.data_loader import get_tasks
from coir.evaluation import COIR
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from tqdm.auto import tqdm
import time
from google import genai
from google.genai import types
from coir.models import YourCustomDEModel

# Tasks to choose from
# ["codetrans-dl", "stackoverflow-qa", "apps","codefeedback-mt", "codefeedback-st", "codetrans-contest", "synthetic-
# text2sql", "cosqa", "codesearchnet", "codesearchnet-ccr"]
TASKS_TO_EVALUATE = ["apps", "stackoverflow-qa", "codetrans-dl", "codefeedback-mt", "codefeedback-st", "codetrans-contest", "synthetic-text2sql", "cosqa", "codesearchnet", "codesearchnet-ccr"]

OPEN_MODELS_TO_EVALUATE = [
    "sentence-transformers/all-MiniLM-L6-v2",
]

GEMINI_MODELS_TO_EVALUATE = [
    "text-embedding-004",
]

BATCH_SIZE = 100  # Default batch size for encoding

class APIModel:
    def __init__(self, model_name, **kwargs):
        # Initialize the voyageai client
        self.go = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.requests_per_minute = 300  # Max requests per minute
        self.delay_between_requests = 60 / self.requests_per_minute  # Delay in seco
        self.model_name = model_name

    def encode_text(self, texts: list, batch_size: int = BATCH_SIZE, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")

        all_embeddings = []
        start_time = time.time()
        # Processing texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            result = self.go.models.embed_content(
                model=self.model_name,
                contents=batch_texts,
                config=types.EmbedContentConfig(task_type=task_type))

            batch_embeddings = []  # Assume the API directly returns embeddings
            for i in range(len(result.embeddings)):
              batch_embeddings.append(result.embeddings[i].values)

            all_embeddings.extend(batch_embeddings)
            # Ensure we do not exceed rate limits
            time_elapsed = time.time() - start_time
            if time_elapsed < self.delay_between_requests:
                time.sleep(self.delay_between_requests - time_elapsed)
                start_time = time.time()

        # Combine all embeddings into a single numpy array
        embeddings_array = np.array(all_embeddings)

        # Logging after encoding
        if embeddings_array.size == 0:
            logging.error("No embeddings received.")
        else:
            logging.info(f"Encoded {len(embeddings_array)} embeddings.")

        return embeddings_array

    def encode_queries(self, queries: list, batch_size: int = BATCH_SIZE, **kwargs) -> np.ndarray:
        #truncated_queries = [query[:256] for query in queries]
        #truncated_queries = ["query: " + query for query in truncated_queries]
        #queries = ["query: "+ query for query in queries]

        query_embeddings = self.encode_text(queries, batch_size, task_type="RETRIEVAL_QUERY")
        return query_embeddings


    def encode_corpus(self, corpus: list, batch_size: int = BATCH_SIZE, **kwargs) -> np.ndarray:
        # texts = [doc['text'][:512]  for doc in corpus]
        # texts = ["passage: " + doc for doc in texts]
        # texts = ["passage: "+ doc['text'] for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        return self.encode_text(texts, batch_size, task_type="RETRIEVAL_DOCUMENT")

for gemini_model in GEMINI_MODELS_TO_EVALUATE:
    print(f"Evaluating Gemini model: {gemini_model}")
    # Initialize the model
    model = APIModel(model_name=gemini_model)

    # Get tasks
    tasks = coir.get_tasks(tasks=TASKS_TO_EVALUATE)

    # Initialize evaluation
    evaluation = COIR(tasks=tasks, batch_size=BATCH_SIZE)

    # Run evaluation
    results = evaluation.run(model, output_folder=f"results/{gemini_model}")
    print(results)

print("\n--- Starting Open Model Evaluations ---")
# --- Step 3: Evaluation Loop ---
for model_name in OPEN_MODELS_TO_EVALUATE:
    safe_model_name = model_name.replace('/', '_')
    print(f"\n{'='*30}\nEvaluating model: {model_name}\n{'='*30}")

    # Load the model
    model = YourCustomDEModel(model_name=model_name)

    tasks = get_tasks(tasks=TASKS_TO_EVALUATE)
    evaluation = COIR(tasks=tasks, batch_size=BATCH_SIZE)
    # Define output folder and run evaluation
    output_folder = f"results/{safe_model_name}"
    os.makedirs(output_folder, exist_ok=True)
    results = evaluation.run(model, output_folder=output_folder)

print("\n✅ All evaluations complete!")

def load_all_results_for_task(task_name, model_list):
    """Loads all result files for a given task from the results directory."""
    all_results = {}
    for model_name in model_list:
        safe_model_name = model_name.replace('/', '_')
        file_path = Path(f"results/{safe_model_name}/{task_name}.json")
        if not file_path.exists():
            print(f"Warning: Result file not found for model '{model_name}' at {file_path}. Skipping.")
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_results[model_name] = data["metrics"]
            print(all_results)
    return all_results

def plot_full_comparison(task_name, model_list):
    """Generates and saves a plot comparing all models across all K values for a task."""
    comparison_data = load_all_results_for_task(task_name, model_list)
    if not comparison_data:
        print(f"No data to plot for task: {task_name}")
        return

    metric_names = list(next(iter(comparison_data.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    fig.suptitle(f'Model Comparison on "{task_name}" Task', fontsize=20, y=0.95)

    all_k_values_across_metrics = set()

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        k_vals_for_metric = set()
        for model_name, metrics in comparison_data.items():
            if metric_name in metrics:
                values_dict, k_values, scores = metrics[metric_name], [], []
                for key, score in values_dict.items():
                    match = re.search(r'\d+$', key)
                    if match:
                        k = int(match.group())
                        k_values.append(k)
                        scores.append(score)
                        k_vals_for_metric.add(k)
                if k_values:
                    sorted_pairs = sorted(zip(k_values, scores))
                    k_vals, score_vals = zip(*sorted_pairs)
                    ax.plot(k_vals, score_vals, marker='o', linestyle='-', label=model_name)
        
        all_k_values_across_metrics.update(k_vals_for_metric)

        ax.set_xscale('log')
        ax.set_title(f'{metric_name} vs. K', fontsize=16)
        ax.set_xlabel('K (cutoff)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.legend(fontsize=10)

    # Standardize x-ticks across all subplots
    sorted_k_vals = sorted(list(all_k_values_across_metrics))
    for ax in axes:
        ax.set_xticks(sorted_k_vals)
        ax.set_xticklabels(sorted_k_vals, rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # --- CHANGE: Save the figure instead of showing it ---
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f"{task_name}_full_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved full comparison plot to: {save_path}")
    plt.close(fig)  # Close the figure to free up memory


def plot_k1_comparison(task_name, model_list):
    """Generates and saves a bar chart comparing all models at K=1."""
    comparison_data = load_all_results_for_task(task_name, model_list)
    if not comparison_data:
        print(f"No data to plot for task: {task_name}")
        return

    metrics_at_1 = ['NDCG@1', 'MAP@1', 'Recall@1', 'P@1']
    plot_data = {model: [] for model in model_list if model in comparison_data}

    for metric_key in metrics_at_1:
        main_metric = metric_key.split('@')[0]
        if main_metric == "P":
            main_metric = "Precision"
        for model_name, all_metrics in comparison_data.items():
            # Check if model_name is a key in plot_data before appending
            if model_name in plot_data:
                score = all_metrics.get(main_metric, {}).get(metric_key, 0)
                plot_data[model_name].append(score)

    fig, ax = plt.subplots(figsize=(15, 8))
    num_models = len(plot_data)
    bar_width = 0.25
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
    ax.legend(title='Models', bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=12)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    
    # Dynamically set y-limit
    max_score = 0
    if plot_data:
        max_score = max(max(s) for s in plot_data.values() if s)
    ax.set_ylim(bottom=0, top=max_score * 1.15 if max_score > 0 else 1)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # --- CHANGE: Save the figure instead of showing it ---
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / f"{task_name}_k1_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved K=1 comparison plot to: {save_path}")
    plt.close(fig) # Close the figure to free up memory

print("\n--- Generating Visualizations ---")
# --- Step 5: Run Visualizations for Each Task ---
for task in TASKS_TO_EVALUATE:
    print(f"\n\n{'#'*40}\n# Visualizing results for: {task}\n{'#'*40}")
    # Plot full comparison
    plot_full_comparison(task, OPEN_MODELS_TO_EVALUATE + GEMINI_MODELS_TO_EVALUATE)
    # Plot K=1 comparison
    plot_k1_comparison(task, OPEN_MODELS_TO_EVALUATE + GEMINI_MODELS_TO_EVALUATE)

print("\n✅ All tasks complete.")