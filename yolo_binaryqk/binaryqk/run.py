"""資料傳送、可重現種子、GPU 占用檢查與原子 JSON 共用工具。"""

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import init_seeds


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class GPUInUseError(RuntimeError):
    """GPU 正由其他工作使用；可等待後重試，不能視為訓練失敗。"""


def require_idle_gpu():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    others = [line.strip() for line in result.stdout.splitlines()
              if line.strip() and line.strip() != str(os.getpid())]
    if others:
        raise GPUInUseError(f"GPU has other compute processes: {others}; do not launch")


def seed(epoch=0):
    init_seeds(epoch, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader(ds, batch, epoch=0):
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, collate_fn=ds.collate_fn,
                      pin_memory=True, generator=torch.Generator().manual_seed(epoch), worker_init_fn=seed_worker)


def transfer(raw, device):
    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
    batch["img"] = batch["img"].float() / 255
    return batch
