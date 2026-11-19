"""Static final queue 使用的 selection／gate／export backend。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .export import export_attention_state, file_sha256
from .final_selection import ZERO_TRAIN_MAP, choose_final, choose_pilot, phase_gate
from .final_workflow import materialize_phase_recipes
from .queue_backend import ResearchQueueBackend
from .queue_model import JobKind, QueueJob, QueueResult, QueueState


class FinalQueueBackend(ResearchQueueBackend):
    def execute(self, job: QueueJob, state: QueueState):
        if job.kind is JobKind.SELECT:
            return self._final_select(job, state)
        if job.id == "export-final":
            return self._export(state)
        if job.id == "export-lr-recovery":
            return self._export_recovery(state, selection_id="lr-recovery-select", export_id=job.id)
        if job.id == "export-recovery":
            return self._export_recovery(state)
        return super().execute(job, state)

    @staticmethod
    def _result(state: QueueState, job_id: str) -> QueueResult:
        result = state.job(job_id).result
        if result is None or result.map50_95 is None or result.checkpoint_path is None:
            raise ValueError(f"selection input {job_id} is incomplete")
        return result

    def _selection_result(
        self, job: QueueJob, winner: str, state: QueueState, details: dict[str, object]
    ) -> QueueResult:
        source = self._result(state, winner)
        run = self.runs_root / job.id / "metrics"
        run.mkdir(parents=True, exist_ok=True)
        path = run / "queue-result.json"
        payload = {
            "map50_95": source.map50_95,
            "map50": source.map50,
            "map75": source.map75,
            "maps": list(source.maps),
            "row_sum_max_error": source.row_sum_max_error,
            "checkpoint_path": source.checkpoint_path,
            "metrics_path": str(path.resolve()),
            "profile_path": source.profile_path,
            "winner": winner,
            **details,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return QueueResult.from_dict(
            {key: value for key, value in payload.items() if key in QueueResult.__dataclass_fields__}
        )

    def _final_select(self, job: QueueJob, state: QueueState) -> QueueResult:
        if job.id == "pilot-select":
            ids = {1e-5: "pilot-1e-5-bittrue", 5e-6: "pilot-5e-6-bittrue"}
            lr = choose_pilot({value: self._result(state, name).map50_95 for value, name in ids.items()})
            materialize_phase_recipes(self.project_root, lr)
            return self._selection_result(job, ids[lr], state, {"selected_lr0": lr})
        if job.id == "lr-block-select":
            from .lr_sweep_workflow import LR_MULTIPLIERS, materialize_lr_recovery_recipes

            winner = max(job.parent_job_ids, key=lambda name: self._result(state, name).map50_95)
            match = re.fullmatch(r"lr-block-(x[124])-bittrue", winner)
            if match is None:
                raise ValueError(f"invalid LR block winner: {winner}")
            label = match.group(1)
            materialize_lr_recovery_recipes(self.project_root, label)
            return self._selection_result(
                job,
                winner,
                state,
                {"selected_multiplier": LR_MULTIPLIERS[label], "selected_label": label},
            )
        if re.fullmatch(r"lr-recovery-(neck|backbone|full)-gate", job.id):
            parent, child = job.parent_job_ids
            parent_result, child_result = self._result(state, parent), self._result(state, child)
            winner = phase_gate(
                parent_id=parent,
                parent_map=parent_result.map50_95,
                child_id=child,
                child_map=child_result.map50_95,
            )
            return self._selection_result(job, winner, state, {"accepted_child": winner == child})
        if job.id == "lr-recovery-select":
            winner = max(job.parent_job_ids, key=lambda name: self._result(state, name).map50_95)
            return self._selection_result(job, winner, state, {"single_trajectory": True, "lr_sweep": True})
        if re.fullmatch(r"recovery-(block|neck|backbone|full)-gate", job.id):
            parent, child = job.parent_job_ids
            parent_result, child_result = self._result(state, parent), self._result(state, child)
            winner = phase_gate(
                parent_id=parent,
                parent_map=parent_result.map50_95,
                child_id=child,
                child_map=child_result.map50_95,
            )
            return self._selection_result(job, winner, state, {"accepted_child": winner == child})
        if job.id == "recovery-select":
            winner = max(job.parent_job_ids, key=lambda name: self._result(state, name).map50_95)
            return self._selection_result(job, winner, state, {"single_trajectory": True})
        if re.fullmatch(r"s[0-2]-phase-[abc]-gate", job.id):
            parent, child = job.parent_job_ids
            parent_result, child_result = self._result(state, parent), self._result(state, child)
            winner = phase_gate(
                parent_id=parent,
                parent_map=parent_result.map50_95,
                child_id=child,
                child_map=child_result.map50_95,
            )
            return self._selection_result(job, winner, state, {"accepted_child": winner == child})
        if re.fullmatch(r"s[0-2]-select", job.id):
            winner = max(job.parent_job_ids, key=lambda name: self._result(state, name).map50_95)
            return self._selection_result(job, winner, state, {"seed": job.id[1]})
        if job.id == "final-select":
            seeds = {name[1]: (name, self._result(state, name).map50_95) for name in job.parent_job_ids}
            decision = choose_final(seeds)
            winner = decision.formal_winner
            if winner == "epoch0-bittrue":
                source_id = winner
            else:
                source_id = winner
            return self._selection_result(job, source_id, state, decision.__dict__)
        raise ValueError(f"unsupported final selection {job.id}")

    def _export(self, state: QueueState) -> QueueResult:
        from ultralytics import YOLO

        selected = self._result(state, "final-select")
        source = Path(selected.checkpoint_path)
        final = self.project_root / "pwl-final-best.pt"
        shutil.copy2(source, final)
        parent = self.project_root / "weights/v1-br-best.pt"
        model = YOLO(str(final)).model
        export_attention_state(
            model,
            self.project_root / "pwl-attention-state.pt",
            parent_sha256=file_sha256(parent),
            final_sha256=file_sha256(final),
        )
        metrics = self.runs_root / "export-final" / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        path = metrics / "queue-result.json"
        payload = {
            "map50_95": selected.map50_95 if selected.map50_95 is not None else ZERO_TRAIN_MAP,
            "map50": selected.map50,
            "map75": selected.map75,
            "maps": list(selected.maps),
            "checkpoint_path": str(final.resolve()),
            "metrics_path": str(path.resolve()),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return QueueResult.from_dict(payload)

    def _export_recovery(
        self, state: QueueState, *, selection_id: str = "recovery-select", export_id: str = "export-recovery"
    ) -> QueueResult:
        from ultralytics import YOLO

        selected = self._result(state, selection_id)
        source = Path(selected.checkpoint_path)
        observed = self.project_root / "pwl-best-observed.pt"
        shutil.copy2(source, observed)
        parent = self.project_root / "weights/v1-br-best.pt"
        model = YOLO(str(observed)).model
        export_attention_state(
            model,
            self.project_root
            / (
                "pwl-lr-recovery-attention-state.pt"
                if export_id == "export-lr-recovery"
                else "pwl-recovery-attention-state.pt"
            ),
            parent_sha256=file_sha256(parent),
            final_sha256=file_sha256(observed),
        )
        metrics = self.runs_root / export_id / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        path = metrics / "queue-result.json"
        payload = {
            "map50_95": selected.map50_95,
            "map50": selected.map50,
            "map75": selected.map75,
            "maps": list(selected.maps),
            "checkpoint_path": str(observed.resolve()),
            "metrics_path": str(path.resolve()),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return QueueResult.from_dict(payload)
