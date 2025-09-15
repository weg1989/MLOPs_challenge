import time, csv, os, psutil, torch, torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
import random

RESULTS_FILE = "results/benchmark_results.csv"

# ---- Preprocessing (ImageNet standard) ----
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ---- Load dataset (subset 5000) ----
val_dataset = ImageNet(root="/home/ghost/densenet_mlop", split="val", transform=val_transform)
indices = random.sample(range(len(val_dataset)), 5000)
subset = Subset(val_dataset, indices)

# ---- Accuracy function ----
def accuracy(output, target, topk=(1,5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res

# ---- System metrics ----
def get_system_metrics(device="cuda"):
    ram_usage = psutil.virtual_memory().used / (1024**2)
    cpu_usage = psutil.cpu_percent(interval=0.1)
    vram_usage, peak_alloc, gpu_util = None, None, None
    if device.startswith("cuda") and torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated() / (1024**2)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    return ram_usage, vram_usage, peak_alloc, cpu_usage, gpu_util

# ---- Benchmark function ----
def run_benchmark(batch_size=8, mode="baseline", device="cuda"):
    # fresh model per run
    model = torchvision.models.densenet121(weights="IMAGENET1K_V1").to(device).eval()

    # apply optimizations
    if mode == "amp_channels_last":
        model = model.to(memory_format=torch.channels_last)
    elif mode == "torchscript":
        example_input = torch.randn(1, 3, 224, 224, device=device)
        model = torch.jit.trace(model, example_input)

    val_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    latencies, correct1, correct5, total = [], 0, 0, 0
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if mode == "amp_channels_last":
                images = images.to(memory_format=torch.channels_last)
            start = time.perf_counter()

            if mode == "amp" or mode == "amp_channels_last":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

            top1, top5 = accuracy(outputs, labels)
            correct1 += top1 * labels.size(0)
            correct5 += top5 * labels.size(0)
            total += labels.size(0)

    latency_ms = sum(latencies) / len(latencies)
    throughput = total / (sum(latencies) / 1000.0)
    top1_acc, top5_acc = 100.0 * correct1 / total, 100.0 * correct5 / total
    ram_usage, vram_usage, peak_alloc, cpu_usage, gpu_util = get_system_metrics(device)
    ram_str = f"{ram_usage:.1f}"
    vram_str = f"{vram_usage:.1f}" if vram_usage else "0.0"
    peak_str = f"{peak_alloc:.1f}" if peak_alloc else "0.0"
    cpu_str = f"{cpu_usage:.1f}"
    gpu_str = f"{gpu_util:.1f}" if gpu_util else "0.0"

    print(
    f"[{mode}] Batch {batch_size}: Lat={latency_ms:.2f} ms, Thr={throughput:.2f} img/s, "
    f"Top1={top1_acc:.2f}%, Top5={top5_acc:.2f}%, "
    f"RAM={ram_str} MB, VRAM={vram_str} MB, Peak={peak_str} MB, CPU={cpu_str}%, GPU={gpu_str}%"
    )

    # save results
    header = [
        "mode", "batch_size", "device", "ram_usage_mb", "vram_usage_mb", "peak_alloc_mb",
        "cpu_utilization_pct", "gpu_utilization_pct", "latency_ms", "throughput_img_s",
        "top1_acc_pct", "top5_acc_pct"
    ]
    row = [
        mode, batch_size, device, ram_usage, vram_usage or 0, peak_alloc or 0,
        cpu_usage, gpu_util or 0, latency_ms, throughput, top1_acc, top5_acc
    ]
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# ---- Main loop ----
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = [1, 4, 8, 16, 32]
    modes = ["baseline", "amp", "amp_channels_last", "torchscript"]

    for mode in modes:
        for bsz in batch_sizes:
            run_benchmark(batch_size=bsz, mode=mode, device=device)
