# scripts

這裡只有薄 wrapper：

- `main.py`：所有命令的統一入口，等價於 `python -m yolo_attention.cli`。
- `smoke_cpu.py`：CPU model construction/forward。
- `train.py`：安全 training 入口；沒有 `--execute` 時只 dry-run。

~~~bash
python scripts/main.py workflow
python scripts/smoke_cpu.py --model yolo26m.yaml --variant configs/variants/h-screen.yaml
python scripts/train.py \
  --variant configs/variants/h-screen.yaml \
  --training configs/training/screening.yaml \
  --run-id h-screen-seed0
~~~

package 必須先安裝。不要在 wrapper 寫模型邏輯或硬編碼資料路徑。
