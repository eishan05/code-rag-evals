# Benchmarking Code RAG in Docker

## Running Benchmarks

### Step 1: cd into the root of this repo

### Step 2: Build the docker image

```shell
docker build -t coir-benchmark .
```

### Step 3: Run the container

Note that you will have to pass in a Gemini API Key.

```shell
docker run \
  --rm \
  -e GOOGLE_API_KEY="YOUR_API_KEY_HERE" \
  -v "$(pwd)/results":/app/results \
  -v "$(pwd)/output_plots":/app/output_plots \
  coir-benchmark
```
>>>>>>> 3290b86 (move code from colab to github)
