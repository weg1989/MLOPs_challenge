# MLOPs_challenge

# DenseNet Benchmarking Suite

This project benchmarks and optimizes **DenseNet** using PyTorch, profiling tools, and Docker.
It evaluates baseline vs. optimized variants across latency, throughput, memory, and accuracy — with results logged to CSV and TensorBoard.

---

## Features

* **DenseNet Benchmarking**

  * Baseline inference profiling
  * Optimizations: AMP (mixed precision), `channels_last`, TorchScript (optional)
* **Metrics Collected**

  * Latency (ms), Throughput (img/s)
  * CPU & GPU utilization (%)
  * RAM & VRAM usage (MB), Peak allocation
  * Accuracy (Top-1, Top-5)
* **Results Management**

  * Benchmark results → `results/benchmark_results.csv`
  * TensorBoard logs → `logs/tensorboard/`
  * Profiling reports → `results/profiles/`
  * Model checkpoints → `results/models/`
* **Containerization**

  * Dockerfile with CUDA 12.1 + cuDNN 8 + Python 3.10
  * Dependencies installed with [uv](https://github.com/astral-sh/uv)
  * TensorBoard exposed on port `6006`
  * Health checks included
* **Automation**

  * `build_and_run.sh` automates build + run
  * Docker Compose orchestrates benchmarking + TensorBoard services

---

## Requirements

* **Docker** with NVIDIA GPU support ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
* **Docker Compose plugin**
* GPU with CUDA 12.1 support (tested on WSL2 Ubuntu 22.04)

---

## Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/weg1989/MLOPs_challenge.git
cd MLOPs_challenge
cd densenet_mlop
```

### 2. Build the Docker image

```bash
docker compose build
```

### 3. Run benchmarks

```bash
docker compose run benchmark
```

Results are saved to:

* `results/benchmark_results.csv` (all metrics)
* `logs/tensorboard/` (TensorBoard logs)

### 4. Start TensorBoard

```bash
docker compose up -d tensorboard
```

Then open: [http://localhost:6006](http://localhost:6006)

### 5. Stop everything

```bash
docker compose down
```

---

## Outputs

* **CSV Results**
  `results/benchmark_results.csv`
  Columns:

  ```
  model_variant,batch_size,device,ram_usage_mb,vram_usage_mb,
  cpu_utilization_pct,gpu_utilization_pct,latency_ms,
  throughput_samples_sec,accuracy_top1,accuracy_top5,
  model_size_mb,optimization_technique
  ```

* **TensorBoard Logs**
  Visualize performance trends (`logs/tensorboard/`).

* **Profiling Reports**
  Detailed execution traces (`results/profiles/`).

---

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── build_and_run.sh
├── requirements.txt
├── scripts/
│   ├── densenet_benchmark_full.py   # Benchmark + profiling script
│   ├── run_all_benchmarks.py        # Runs baseline + optimizations
├── results/                         # CSV + profiles + models
├── logs/                            # TensorBoard logs
└── README.md
```

---
