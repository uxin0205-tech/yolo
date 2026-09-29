# Final 交付區

> **Cleanup-v1 發行範圍：** 本分支保留程式、設定、報告、統計圖表與 manifests；checkpoint、資料集影像／標註、逐筆預測、batch 預覽和冗長執行日誌已從此發行快照排除。下文的歷史權重路徑、數量、checksums 與實驗結果仍保留作研究紀錄；訓練、推論、完整交付驗證及資料重建需要另行提供原始資產，不代表 clone 後即可直接重跑。 詳見[清理範圍與資產需求](../../docs/cleanup-v1/README.md)。


- `full35/`：J3正式Full35 shared-trunk Detect/Pose交付包。
- `full35-j2-archive/`：升格前完整J2 package，待另行授權清理。
- Partial75 尚未執行，不得混入 Full35；未來若啟動會另建 `partial75/`。

## GitHub 與權重

依使用者2026-08-27授權，這個`final/`會完整發布，包含J3主權重、J2 rollback／archive、
獨立Detect/Pose baseline與產圖所需證據。所有`.pt`由本目錄`.gitattributes`強制使用
Git LFS；clone前必須先安裝`git-lfs`，clone後執行`git lfs pull`取得權重內容。

目前共有34個`.pt`路徑引用；相同內容由LFS依OID去重後為17個物件、
3,613,842,852 bytes。`__pycache__`與`.pyc`是可重建cache，不屬package manifest，
不會提交GitHub，但本次沒有從本機刪除。
