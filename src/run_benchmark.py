import coir
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

class APIModel:
    def __init__(self, **kwargs):
        # Initialize the voyageai client
        self.go = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.requests_per_minute = 300  # Max requests per minute
        self.delay_between_requests = 60 / self.requests_per_minute  # Delay in seco

    def encode_text(self, texts: list, batch_size: int = 100, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")

        all_embeddings = []
        start_time = time.time()
        # Processing texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            print("batch:", {i}, len(batch_texts))
            result = self.go.models.embed_content(
                model="text-embedding-004",
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
        print(embeddings_array.shape)

        # Logging after encoding
        if embeddings_array.size == 0:
            logging.error("No embeddings received.")
        else:
            logging.info(f"Encoded {len(embeddings_array)} embeddings.")

        return embeddings_array

    def encode_queries(self, queries: list, batch_size: int = 100, **kwargs) -> np.ndarray:
        #truncated_queries = [query[:256] for query in queries]
        #truncated_queries = ["query: " + query for query in truncated_queries]
        #queries = ["query: "+ query for query in queries]

        query_embeddings = self.encode_text(queries, batch_size, task_type="RETRIEVAL_QUERY")
        return query_embeddings


    def encode_corpus(self, corpus: list, batch_size: int = 100, **kwargs) -> np.ndarray:
        # texts = [doc['text'][:512]  for doc in corpus]
        # texts = ["passage: " + doc for doc in texts]
        # texts = ["passage: "+ doc['text'] for doc in corpus]
        texts = [doc['text'] for doc in corpus]
        return self.encode_text(texts, batch_size, task_type="RETRIEVAL_DOCUMENT")

# Load the model
model = APIModel()

# Get tasks
#all task ["codetrans-dl", "stackoverflow-qa", "apps","codefeedback-mt", "codefeedback-st", "codetrans-contest", "synthetic-
# text2sql", "cosqa", "codesearchnet", "codesearchnet-ccr"]
tasks = coir.get_tasks(tasks=["codetrans-dl"])

# Initialize evaluation
evaluation = COIR(tasks=tasks, batch_size=100)

# Run evaluation
results = evaluation.run(model, output_folder=f"results/gemini")
print(results)

# For open source models, you can use the following code snippet
# Models and tasks to evaluate
MODELS_TO_EVALUATE = [
    "sentence-transformers/all-MiniLM-L6-v2",
]
TASKS_TO_EVALUATE = ["apps"]

print("\n--- Starting Model Evaluations ---")
# --- Step 3: Evaluation Loop ---
evaluation_results = {}
for model_name in MODELS_TO_EVALUATE:
    safe_model_name = model_name.replace('/', '_')
    print(f"\n{'='*30}\nEvaluating model: {model_name}\n{'='*30}")

    # Load the model
    model = YourCustomDEModel(model_name=model_name)

    for task_name in TASKS_TO_EVALUATE:
        print(f"\n--- Running task: {task_name} ---")

        # Get the specific task
        tasks = get_tasks(tasks=[task_name])
        evaluation = COIR(tasks=tasks, batch_size=100)

        # Define output folder and run evaluation
        output_folder = f"results/{safe_model_name}"
        os.makedirs(output_folder, exist_ok=True)
        results = evaluation.run(model, output_folder=output_folder)

        print(f"--- Finished task: {task_name} ---")
        evaluation_results.update(results)

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
    """Generates and shows a plot comparing all models across all K values for a task."""
    comparison_data = load_all_results_for_task(task_name, model_list)
    if not comparison_data:
        print(f"No data to plot for task: {task_name}")
        return

    metric_names = list(next(iter(comparison_data.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    fig.suptitle(f'Model Comparison on "{task_name}" Task', fontsize=20, y=0.95)

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        for model_name, metrics in comparison_data.items():
            if metric_name in metrics:
                values_dict, k_values, scores = metrics[metric_name], [], []
                for key, score in values_dict.items():
                    match = re.search(r'\d+$', key)
                    if match:
                        k_values.append(int(match.group()))
                        scores.append(score)
                if k_values:
                    sorted_pairs = sorted(zip(k_values, scores))
                    k_vals, score_vals = zip(*sorted_pairs)
                    ax.plot(k_vals, score_vals, marker='o', linestyle='-', label=model_name)

        all_k = sorted(list(set(k_vals)))
        ax.set_xscale('log')
        ax.set_xticks(all_k)
        ax.set_xticklabels(all_k, rotation=45)
        ax.set_title(f'{metric_name} vs. K', fontsize=16)
        ax.set_xlabel('K (cutoff)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, which="both", ls="--", c='0.7')
        ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def plot_k1_comparison(task_name, model_list):
    """Generates and shows a bar chart comparing all models at K=1."""
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

    ax.set_title(f'Model Comparison at K=1 for "{TASK_TO_VISUALIZE}" Task', fontsize=18, pad=20)
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_at_1, fontsize=12)
    ax.legend(title='Models', bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=12)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.set_ylim(bottom=0, top=max(max(s) for s in plot_data.values()) * 1.15 if plot_data else 1)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

print("\n--- Generating Visualizations ---")
# --- Step 5: Run Visualizations for Each Task ---
for TASK_TO_VISUALIZE in TASKS_TO_EVALUATE:
    print(f"\n\n{'#'*40}\n# Visualizing results for: {TASK_TO_VISUALIZE}\n{'#'*40}")
    plot_full_comparison(TASK_TO_VISUALIZE, MODELS_TO_EVALUATE)
    plot_k1_comparison(TASK_TO_VISUALIZE, MODELS_TO_EVALUATE)

print("\n✅ All tasks complete.")