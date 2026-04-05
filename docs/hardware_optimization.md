# Hardware Optimization Guide for AI on Old Laptops

## The Goal

Extract maximum AI performance from CPU-only hardware through software and
configuration changes — no hardware upgrades required for most optimizations.

---

## 1. CPU Governor (Free Performance)

The Linux CPU governor controls power/performance tradeoff. Setting it to `performance`
disables throttling and runs the CPU at maximum frequency during inference.

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set performance mode (requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# Install cpupower for persistent setting
sudo apt install linux-tools-common
sudo cpupower frequency-set -g performance
```

**Expected gain:** 15–40% faster inference on thermally-throttled laptops.

---

## 2. Thread Tuning for Ollama / llama.cpp

By default, Ollama uses all available threads. For hyperthreaded CPUs (most i3/i5/i7),
using only physical cores (not hyperthreads) often gives better performance:

```bash
# Check physical cores vs logical cores
lscpu | grep -E "Core|Thread|CPU\(s\)"

# Set for Ollama (add to ~/.bashrc or run before ollama serve)
export OLLAMA_NUM_PARALLEL=1
export OMP_NUM_THREADS=2  # set to physical core count

# For llama.cpp directly
./main -t 2  # -t = thread count
```

---

## 3. RAM Management

Close memory hogs before running LLM inference:

```bash
# Check current RAM usage
free -h
htop  # or btop for a nicer view

# RAM consumers to close:
# - Web browser tabs (each can use 100-300MB)
# - Electron apps (VS Code, Discord, Slack)
# - Multiple terminal sessions

# Check which processes use the most RAM
ps aux --sort=-%mem | head -15

# Drop filesystem cache (frees RAM without rebooting)
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
```

**Rule of thumb:** Leave at least 2GB free for the model to load comfortably.

---

## 4. Lightweight Desktop Environment

Switching from GNOME to XFCE or LXQt saves significant RAM:

| DE | Idle RAM | Notes |
|----|----------|-------|
| GNOME | ~800MB | Default Ubuntu |
| KDE Plasma | ~600MB | Feature-rich |
| XFCE | ~280MB | Recommended (Xubuntu) |
| LXQt | ~200MB | Lightest full DE |
| i3/sway | ~80MB | Tiling WM (advanced) |

```bash
# Install XFCE on Ubuntu
sudo apt install xubuntu-desktop

# At login screen, click the gear icon to select XFCE
```

---

## 5. Swap Configuration

For models that slightly exceed physical RAM, swap can help (but is slow on HDD):

```bash
# Check current swap
swapon --show

# Create a swap file (4GB example)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Tune swappiness (lower = use swap less)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

**On SSD:** Swap is acceptable. **On HDD:** Avoid LLMs that need swap.

---

## 6. Model Selection Guide

Choose the right model for your RAM:

| Available RAM | Recommended Model | Format | Speed |
|---------------|------------------|--------|-------|
| 3 GB | TinyLlama 1.1B | Q4_K_M | Fast |
| 4 GB | Qwen2 1.5B | Q4_K_M | Good |
| 6 GB | Gemma2 2B | Q4_K_M | Excellent |
| 8 GB | Phi-3 Mini 3.8B | Q4_K_M | Great |
| 12 GB | Llama 3.2 3B | Q4_K_M | Very good |

---

## 7. Thermal Management

Sustained AI workloads generate sustained heat. Laptops throttle when too hot:

```bash
# Monitor CPU temperature
sudo apt install lm-sensors
sensors

# Or watch continuously
watch -n 1 sensors

# Check throttling events
sudo apt install stress
dmesg | grep -i throttl

# Practical tips:
# - Use on a hard, flat surface (not bed/carpet)
# - Point a small USB fan at the ventilation
# - Clean dust from vents periodically
# - Use laptop cooler stands
```

---

## 8. SSD vs HDD Impact

An SSD dramatically improves AI workflow performance:

| Operation | HDD | SSD | Speedup |
|-----------|-----|-----|---------|
| Model load (2GB) | ~40s | ~3s | 13x |
| Chroma DB index | ~30s | ~2s | 15x |
| Python import | ~5s | ~0.5s | 10x |
| Dataset load | ~10s | ~0.8s | 12x |

If upgrading hardware is an option, an SSD is the highest-ROI upgrade for AI work.

---

## 9. Monitoring Dashboard

```bash
# Install monitoring tools
sudo apt install btop htop iotop

# Run during inference to see CPU/RAM/IO usage
btop

# Log performance metrics
sar -u 1 60 > cpu_log.txt  # CPU usage every second for 1 minute
```

---

## 10. Quick Performance Checklist

Before running heavy AI workloads:

- [ ] CPU governor set to `performance`
- [ ] Browser closed (or tabs minimized)
- [ ] Unnecessary apps closed
- [ ] `free -h` shows > 2GB available
- [ ] Laptop on hard surface
- [ ] Temperature < 85°C (`sensors`)
- [ ] Swap configured (if RAM < 8GB)
- [ ] Using Q4_K_M quantized model
