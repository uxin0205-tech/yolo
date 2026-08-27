# Final 交付區

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
