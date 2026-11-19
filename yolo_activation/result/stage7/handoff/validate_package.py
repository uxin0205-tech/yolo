"""在新程序核對可攜載入與原模型逐 tensor 等價；不使用 GPU／資料集。"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_CONFIG_DIR"] = tempfile.mkdtemp(prefix="stage7-yolo-config-")


def main():
    import shutil

    import torch
    from load_model import canonical, load_model, tensors, verify_contract

    torch.set_num_threads(2)
    # .venv 的第三方套件允許；舊研究程式、權重與資料集一律禁止讀取。
    environment = json.loads((HERE / "environment.json").read_text())
    workspace = Path(environment["runtime_source"]).parents[1]
    denied_reads = []
    guard_enabled = True

    def guard(event, args):
        if not guard_enabled:
            return
        if event == "socket.connect":
            raise PermissionError("交付驗證不允許網路連線")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if (path.is_relative_to(workspace) and not path.is_relative_to(HERE)
                    and not path.is_relative_to(workspace / ".venv")):
                denied_reads.append(str(path))
                raise PermissionError(f"交付包讀取研究目錄：{path}")

    sys.addaudithook(guard)
    reference = json.loads((HERE / "validation/reference.json").read_text())
    results = {}
    try:
        for candidate, sizes in reference["models"].items():
            model = load_model(candidate=candidate)
            row = {"contract": verify_contract(model, canonical(model.stage7_manifest)), "sizes": {}}
            for size_text, expected in sizes.items():
                size = int(size_text)
                torch.manual_seed(reference["seed"] + size)
                inputs = torch.rand(1, 3, size, size)
                with torch.inference_mode():
                    outputs = model(inputs, task="both")
                    actual = {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                                     "sha256": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()}
                              for name, t in tensors(outputs)}
                    if actual != expected:
                        raise ValueError(f"{candidate}/{size} 與原模型輸出不相同")
                    for task in ("detect", "pose"):
                        single = model(inputs, task=task)
                        if set(single) != {task}:
                            raise ValueError("單任務 routing 錯誤")
                        for (_, a), (_, b) in zip(tensors(single[task]), tensors(outputs[task]), strict=True):
                            if not torch.equal(a, b):
                                raise ValueError("單任務／雙任務輸出不同")
                row["sizes"][size_text] = {"tensors": len(actual), "bitwise_equal": True,
                                            "max_abs_error": 0.0, "routing_equal": True}
                print(f"PASS {candidate} {size}: {len(actual)} tensors 完全相同", flush=True)
            manifest = canonical(model.stage7_manifest)
            first = next(iter(manifest["activation"]["sites"]))
            original = model.get_submodule(first)
            model.set_submodule(first, torch.nn.SiLU())
            try:
                verify_contract(model, manifest)
            except ValueError:
                row["wrong_activation_rejected"] = True
            else:
                raise AssertionError("錯誤 activation 未拒絕")
            model.set_submodule(first, original)
            normalizer = model.get_submodule("graph.model.10.m.0.attn.normalizer")
            buffer = next(iter(normalizer.state_dict().values()))
            saved = buffer.clone()
            buffer.reshape(-1)[0] += 1
            try:
                verify_contract(model, manifest)
            except ValueError:
                row["wrong_constant_rejected"] = True
            else:
                raise AssertionError("錯誤 PWL／reciprocal 常數未拒絕")
            buffer.copy_(saved)
            results[candidate] = row
            del model, outputs, single
        if denied_reads:
            raise ValueError(f"嘗試依賴研究目錄：{denied_reads}")
        result = {"status": "passed", "scope": "CPU FP32 160／640 合成輸入、原模型逐值等價、routing、契約負向檢查；不是重跑 AP", "isolated_process": sys.flags.isolated == 1,
                  "workspace_reads_denied": True, "external_workspace_reads": denied_reads,
                  "gpu_used": False, "models": results}
        (HERE / "validation/result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    finally:
        # shutil 使用 dir_fd 開啟相對目錄；模型檢查完成後才解除研究目錄阻擋。
        guard_enabled = False
        shutil.rmtree(os.environ["YOLO_CONFIG_DIR"])


if __name__ == "__main__":
    main()
