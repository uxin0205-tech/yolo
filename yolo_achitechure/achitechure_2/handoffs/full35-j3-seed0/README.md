# Full35 J3 handoff

這個資料夾是 architecture_2 對 `/home/uxin/yolo/yolo_combine/final/full35/` 的不可變接收契約；
它不複製模型程式或權重，也不使用 `latest` 類型的可變路徑。

## 檔案

| 檔案 | 用途 |
|---|---|
| [manifest.json](manifest.json) | checkpoint、builder、資料、graph、protected paths與全部SHA256 |
| [training-recipe.yaml](training-recipe.yaml) | C0～C3 Float20共用的MuSGD、batch、LR、loss、augmentation與freeze recipe |
| [fresh-process-report.json](fresh-process-report.json) | 本專案對J3重新產生的CPU strict-load與Detect／Pose forward證據 |

真正 checkpoint 固定在上游 release：
`/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`。

## 固定契約

- revision：`full35-final-j3-seed0-d67fb45c`。
- parent：accepted J3 EMA；J2只保留於上游rollback，不參與本輪。
- checkpoint SHA256：`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- model：23層shared graph + Detect/Pose雙head。
- 輸出：COCO80 Detect、BBAT5 ball/bat Pose、`kpt_shape=[2,3]`。
- 候選路徑：`graph.model.6/8/13/19`；只允許C0～C3單因子。
- 凍結：inherited MASF與三個attention paths；兩個heads受契約保護。

上游若再升格其他winner或改動checkpoint，必須建立新的handoff資料夾、revision與hash；不得覆寫本目錄
或把不同parent的結果混在同一Float20矩陣。

## 驗證

    export PYTHONPATH="$PWD/src"
    PY=/home/uxin/yolo/.venv/bin/python
    $PY -m achitechure_2 inspect-handoff --manifest handoffs/full35-j3-seed0/manifest.json
    $PY -m achitechure_2 accept-handoff --manifest handoffs/full35-j3-seed0/manifest.json --model-loader achitechure_2.full35_adapter:load_full35_parent

`inspect-handoff`只驗metadata；`accept-handoff`會真正重建graph、嚴格載入1,238 tensors並驗證
Candidate/protected/frozen paths。驗收產物寫入`artifacts/intake/accepted.json`，可重建且不提交。

## Git 邊界

只提交本README、manifest、training recipe與fresh-process report。checkpoint、`.pt`、上游source bundle、cache與run都不複製
進architecture_2；manifest中的絕對路徑是本機handoff引用，不代表Git會上傳該權重。
