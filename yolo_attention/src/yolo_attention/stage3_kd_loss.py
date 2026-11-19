"""optimize spatial attention transfer；GT bat exclusion 是本專案延伸。"""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .config import Stage3KDConfig


@contextmanager
def capture(layers):
    values, handles = {}, []
    try:
        for i in Stage3KDConfig().sites:
            def hook(module, args, output, index=i):
                if index in values:
                    raise RuntimeError('repeated KD feature site')
                values[index] = output
            handles.append(layers[i].register_forward_hook(hook))
        yield values
    finally:
        for handle in handles:
            handle.remove()
        values.clear()


def protected_mask(batch, shape, config):
    b, _, h, w = shape
    boxes = batch['bboxes']
    classes = batch['cls'].reshape(-1)
    indices = batch['batch_idx'].reshape(-1)
    if boxes.shape != (len(classes), 4) or len(indices) != len(classes):
        raise ValueError('KD expects native normalized xywh labels')
    if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
        raise ValueError('KD boxes must be finite normalized xywh')
    selected = classes == config.excluded_class
    mask = torch.ones((b, h, w), dtype=torch.bool, device=boxes.device)
    for index, box in zip(indices[selected].tolist(), boxes[selected].tolist()):
        if int(index) != index or not 0 <= index < b:
            raise ValueError('KD label batch index outside images')
        x, y, bw, bh = box
        halo = config.halo_cells
        x0 = max(0, math.floor((x-bw/2)*w)-halo)
        x1 = min(w, math.ceil((x+bw/2)*w)+halo)
        y0 = max(0, math.floor((y-bh/2)*h)-halo)
        y1 = min(h, math.ceil((y+bh/2)*h)+halo)
        mask[int(index), y0:y1, x0:x1] = False
    return mask


def spatial_loss(student, teacher, batch, config):
    if len(student) != 3 or len(teacher) != 3:
        raise ValueError('KD requires P3/P4/P5')
    terms = []
    for s, t in zip(student, teacher):
        if s.ndim != 4 or t.ndim != 4 or s.shape[0] != t.shape[0] or s.shape[2:] != t.shape[2:]:
            raise ValueError('KD requires identical spatial coordinates')
        mask = protected_mask(batch, s.shape, config)
        sa = F.normalize((s.float().square().mean(1)*mask).flatten(1), p=2, dim=1, eps=1e-6)
        ta = F.normalize((t.detach().float().square().mean(1)*mask).flatten(1), p=2, dim=1, eps=1e-6)
        terms.append((sa-ta).square().sum(1))
    result = torch.stack(terms).mean(0)
    if not torch.isfinite(result).all():
        raise FloatingPointError('nonfinite spatial KD')
    return result


def parts(model, teacher, batch):
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('KD teacher must be eval and frozen')
    with capture(teacher.model) as tf:
        with torch.no_grad():
            teacher(batch['img'])
        with capture(model.model) as sf:
            native, items = model.loss(batch)
            features = [sf[i] for i in Stage3KDConfig().sites]
            kd = spatial_loss(features, [tf[i] for i in Stage3KDConfig().sites], batch, Stage3KDConfig()).sum()
    return native.sum(), kd, features, items
