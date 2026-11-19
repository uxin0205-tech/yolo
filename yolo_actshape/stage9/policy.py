"""可測試的條件 queue 選型，不以未改方法 control 加訓擴張矩陣。"""


def choose_training(candidates, results, base, limits, scoring):
    picked, reasons = [], {}
    for group in ("efficient", "quality"):
        eligible = []
        for name, spec in candidates.items():
            if not spec["train"] or spec["kind"] != group:
                continue
            values = results[name]["metrics"]
            drops = {k: values[k] - v for k, v in base.items()}
            if min(drops.values()) < -limits["screen_max_any_drop"] or scoring(values) < scoring(base) + limits["screen_min_score_delta"]:
                reasons[name] = {"train": False, "reason": "E0 退化超過預先設定的恢復範圍", "deltas": drops}
            else:
                eligible.append(name)
        if eligible:
            selected = max(eligible, key=lambda n: scoring(results[n]["metrics"]))
            picked.append(selected)
            for name in eligible:
                reasons[name] = {"train": name == selected, "reason": "同成本組 E0 平衡分數選一個"}
    return picked[:limits["max_train_candidates"]], reasons
