"""單一 Stage6 入口；背景 queue/狀態/受監督 worker。"""

import argparse
import json
import traceback
from pathlib import Path

from .common import ARTIFACTS, read, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status", "worker"))
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "rebuild-20261029")
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        state = read(args.output / "supervisor.json")
        if state.get("job"):
            path = args.output / state["job"] / "progress.json"
            if path.exists():
                state["latest_progress"] = read(path)
        print(json.dumps(state, ensure_ascii=False, indent=2))
    elif args.command == "run":
        from .runner import run

        run(args.output)
    else:
        if args.spec is None:
            parser.error("worker requires --spec")
        from .runner import worker

        spec = read(args.spec)
        try:
            worker(spec)
        except BaseException:
            write_json(
                Path(spec["output"]) / "error.json",
                {"traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
