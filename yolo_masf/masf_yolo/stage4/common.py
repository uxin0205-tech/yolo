"""固定來源、事前配方與小型結果工具。"""

import json
from pathlib import Path

from binaryqk.model import sha256
from binaryqk.run import write_json

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'yolo_masf'
ARTIFACTS = PROJECT / 'artifacts/stage4'
REVIEW = ARTIFACTS / 'reference-review.json'
KD = ROOT / 'yolo_attention/result/stage3/kd-epoch05'
KD_RUN = ROOT / 'yolo_attention/artifacts/stage3/kd-20261019-v1'


def read(path):
    return json.loads(Path(path).read_text())


def verify_reference():
    data = read(REVIEW)
    if not data['reviewed_before_implementation']:
        raise ValueError('missing prior optimize review')
    for path, digest in data['immutable_reference_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'reference changed: {path}')
    return data


def parent(task):
    data = verify_reference()
    if task == 'detect':
        path = KD / 'evaluated.pt'
        digest = read(KD / 'export.json')['checkpoint_sha256']
    elif task == 'pose':
        path, digest = Path(data['pose_parent']), data['pose_parent_sha256']
    else:
        raise ValueError(task)
    if sha256(path) != digest:
        raise ValueError(f'{task} parent SHA mismatch')
    return path, digest


def emit(output, event):
    import time
    event = {'time_unix': time.time(), **event}
    write_json(Path(output) / 'progress.json', event)
    with (Path(output) / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
