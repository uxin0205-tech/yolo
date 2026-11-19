# YOLO 棒球偵測與姿態研究

本研究把 Detect 與 Pose 合併為一個模型，再依序驗證算子簡化、RepConv 折疊與輸入形狀。第 1～9 階段均已完成。依 11 月 19 日決定，交付兩個有各自權重的部署版本。尺寸一律按 **高 × 寬（H×W）** 表示。

| 選擇 | 輸入與權重 | COCO AP | COCO 球棒 AP | BBAT 框 AP | BBAT Pose AP |
|---|---|---:|---:|---:|---:|
| [硬體用 512×768](result/hardware/README.md) | 固定 512×768，形狀適應第 2 輪 | 51.9632% | 45.3351% | 62.4501% | 89.9026% |
| [精度用 rect640](yolo_actshape/result/stage9/handoff/README.md) | 隨批次補邊，qSiLU Rep17＋20 第 1 輪 | 51.4347% | 46.9576% | 61.4371% | 90.1389% |

兩列均為 AP50–95，使用 COCO val2017 的 5,000 張與 BBAT5 v1 驗證集的 683 張。固定尺寸版的 COCO 整體與 BBAT 框較高，原 rect640 的 COCO 球棒與 BBAT Pose 較高。兩個版本都保有 qSiLU、B2 二值 QK、S1 Q10 scale 與 signed16 bias、D1 整數 PWL exp 與 LUT256 reciprocal，以及 folded Rep17＋20。[模型載入與十項 AP](result/README.md) · [選型 manifest](result/manifest.json)

## 九個實驗步驟

同一階段的候選從共同起點分開比較。下表列出每一步要回答的問題，以及實際交給下一步的模型。

| 步驟 | 研究問題 | 結果與下一步的起點 |
|---|---|---|
| 1 · [Binary QK](yolo_binaryqk/result/stage1/README.md) | 既有 Detect 的兩處 Attention 如何用二值分支估計 QK 分數？ | 比較 residual、Hadamard 與三分支，選 B2 原 sign＋Hadamard。 |
| 2 · [scale／bias](yolo_binaryqk/result/stage2/README.md) | 分數縮放適合固定 Q10 或二次冪？ | 從 B2 分別訓練，選 S1 Q10 與 signed16 水平／垂直相對位置 bias。 |
| 3 · [Softmax](yolo_attention/result/stage3/README.md) | 怎麼近似 exp 和每列的除法？ | 選 D1 整數 PWL exp（−10 至 0，20 段）＋256 項倒數表；知識蒸餾（KD）恢復到 E5。 |
| 4 · [MASF](yolo_masf/artifacts/stage4/extension-20261019-v1/RESULTS.md) | 在 Detect 與已訓 Pose 各自的 P3 加 MASF，球與棒是否受益？ | 兩任務各自評估後皆沿用原模型。候選及數據留在原資料夾。 |
| 5 · [HOG](yolo_hog/result/stage5/README.md) | 訓練時加入邊緣方向監督，能改善 Detect 嗎？ | 恢復結果未支持採用，帶第 3 步的 Detect 進融合。 |
| 6 · [Detect–Pose 融合](yolo_combine/result/stage6/README.md) | 如何共享 trunk，保存兩個任務的精度？ | 接入已訓 Pose，分段調整共享層與 heads。選 KD E2 接 Pose head 恢復 E1。 |
| 7 · [Activation](yolo_activation/result/stage7/README.md) | 共享模型各區域對 activation 多敏感？ | 比較 qSiLU、Hardswish、PolyShift 與 PolyQuality。PolyQuality E10 綜合最佳，選 qSiLU E8 交給 RepConv。 |
| 8 · [RepConv／ACB](yolo_achitechure/result/stage8-retrain/README.md) | 哪個位置與分支可在部署時折疊？ | 選 **qSiLU 訓練的 Rep17＋20 E1**。訓練用 Float PWL，部署用 D1；第 17、20 層各折成一個 3×3 Conv，之後接 qSiLU。 |
| 9 · [yolo_reshape 特徵圖研究](yolo_reshape/README.md) | 改成長方形或二次冪後，每層尺寸、精度與成本怎麼變？ | 在原工作區 `yolo_actshape/` 跑完六種尺寸、每種 20 輪。交付固定 512×768 硬體版與原 rect640 精度版。 |

## yolo_reshape 與硬體選型

**`yolo_reshape/` 是特徵圖研究的導讀入口。** 原程式、protocol、逐尺寸 AP、GPU 計時與分塊驗證均在 `yolo_actshape/`。固定 512×768 的五級特徵圖依序為 256×384、128×192、64×96、32×48、16×24，硬體可以預先確定每一列要存幾筆。原 rect640 保持影像比例，依批次補齊尺寸，特徵圖的高、寬也隨批次改變。[形狀比較與硬體邊界](yolo_reshape/README.md)

## 工作區目錄

| 要找的內容 | 從這裡進入 |
|---|---|
| 兩個部署模型、十項 AP、載入命令與版本指紋 | [result/](result/README.md) · [選型 manifest](result/manifest.json) |
| 各階段的實作、設定與評估 | 依上表 1～9 的連結進入各研究資料夾。形狀導讀在 [yolo_reshape/](yolo_reshape/README.md)，實作與完整結果在 [yolo_actshape/](yolo_actshape/README.md) |
| 原始訓練來源與資料配置 | [資料 registry](configs/datasets/bbat5-v1.yaml) · [BBAT5 規範](docs/agents/bbat5-datasets.md) · [COCO 配置](coco2017.yaml) · [已訓 Pose](yolo_pose_training/README.md) |
| 全輪結果、計畫與可重畫數據 | [研究總報告](docs/research/yolo-research-summary-2026-11-17.md) · [研究索引](docs/research/README.md) · [整合計畫](docs/research/yolo-research-plan-2026-09-21.md) · [數據與繪圖](docs/research/data/integrated-20261117/README.md) |
| 歷史對照與額外研究 | [舊 Attention 交付](yolo_attention_final/README.md) · [P2 實驗](yolo_p2/P2_PLAN.md) · [量化研究](yolo_quantize/README.md) · [清理計畫沿革](yolo_cleanup.md) |
| 工作紀錄、開發與執行環境 | [工作紀錄索引](docs/worklogs/README.md) · [工作區規則](AGENTS.md) · [執行相依](requirements.txt) · [開發相依](requirements-dev.txt) |
| 個人簡報與 PDF | 本機 `docs/local-materials/` |

BBAT5 v1 的正式配置和指定的影像切分以 [Detect YAML](original/pose/derived/bbat5-v1/configs/detect.yaml) 與 [Pose YAML](original/pose/derived/bbat5-v1/configs/pose.yaml) 為準。本機資料放在 `original/`、`coco2017/`，`pose_dataset` 是指向 `original/pose/` 的舊路徑相容連結。`.venv/` 是本機 Python 環境。

## GitHub 交付範圍

GitHub `main` 收錄實驗程式、配置、報告、小型數據與模型載入契約。完整模型權重、原始資料、執行快取與本機文件由工作站保存。部署時依 SHA-256 放回對應權重。[權重位置與載入方法](result/README.md)

`yolo_optimize/` 為已退役的舊整合目錄。舊報告中的路徑用於追溯當時實驗，兩個部署入口與整輪小型結果可獨立使用。[舊目錄清理與驗證](docs/worklogs/2026-11-19-selections-and-cleanup.md) · [本次整理與 GitHub 交付](docs/worklogs/2026-11-20-workspace-github.md)
