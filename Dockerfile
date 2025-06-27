# Use an official Python 3.10 slim runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container from the build context root
COPY requirements.txt .

# Install Python Essentials
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
        build-essential checkinstall libffi-dev python-dev-is-python3 \
        libncursesw5-dev libssl-dev \
        libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies using pip
# --no-cache-dir reduces image size by not storing the pip cache
RUN pip install --no-cache-dir -r requirements.txt

# --- UPDATED LINE ---
# Copy the main benchmark script from the src directory into the container's WORKDIR
COPY src/run_benchmark.py .

# Create directories for the output results and plots
# The script will write into these directories inside the container.
RUN mkdir -p /app/results /app/output_plots

# Set the default command to run when the container starts
CMD ["python", "run_benchmark.py"]