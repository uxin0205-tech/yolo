---
status: accepted
---

# 發布 original 可追溯資料但永久排除權重

依使用者 2026-08-23 明確授權，GitHub 發布 `original/pose/dataset/`、
`detect_dataset/` 與 `derived/bbat5-v1/`，讓 raw、legacy 與 canonical lineage 可被完整稽核。
這項發布不把 raw/basic split 升格為正式訓練入口；新 run 仍固定使用 `bbat5-v1` registry。
權重、checkpoint、cache、run 與 deployment artifact 永久排除，遠端 tree 中既有的兩個
original `.pt` 也移除。`detect_dataset.zip` 超過 GitHub 一般 Git 單檔限制且與完整解壓目錄
重複，因此只發布解壓內容。canonical image links 在 Git tree 改用 repository-relative symlink，
避免 clone 後指向不存在的 `/home/uxin/...`。
