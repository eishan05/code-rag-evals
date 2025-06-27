# Benchmarking Code RAG in Docker

## Running Benchmarks

### Step 1: `cd` into the root of this repo

### Step 2: Build the docker image

```shell
sudo docker build -t coir-benchmark .
```

### Step 3: Run the container

Note that you will have to pass in a Gemini API Key.

```shell
sudo docker run \
  --rm \
  -e GOOGLE_API_KEY="YOUR_API_KEY_HERE" \
  -v "$(pwd)/results":/app/results \
  -v "$(pwd)/plots":/app/plots \
  coir-benchmark
```
