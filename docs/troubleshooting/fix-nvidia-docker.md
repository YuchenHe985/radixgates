# Fix NVIDIA GPU Not Accessible Inside Docker

## Symptom

`nvidia-smi` works on the host, but:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# Error: could not select device driver "" with capabilities: [[gpu]]
```

Or SGLang containers start but fail with CUDA initialization errors.

## Fix

Install and configure `nvidia-container-toolkit`:

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --batch --yes --dearmor \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Register NVIDIA runtime with Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to pick up the new runtime
sudo systemctl restart docker
```

Verify:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# Must show GPU table with your RTX 4090s
```
