# 2026-08-27 GitHub Full35 final發布

## 變更內容與原因

依使用者明確要求，將`yolo_combine/final/`完整發布到
`uxin0205-tech/yolo`的`main`，commit訊息固定為`5090 Full35 0827`。發布範圍保留：

- `final/full35/`的J3主交付、J2 rollback、獨立baseline、程式快照、分析與圖；
- `final/full35-j2-archive/`的完整J2回退package；
- 所有產圖與復現所需`.pt`權重。

遠端原本沒有`yolo_combine/final/`。父repository本地工作樹ahead 1、behind 46且含大量
使用者未追蹤內容，因此發布從最新`origin/main`建立隔離worktree，只複製授權範圍，
不提交或覆寫父工作樹其他變更。

遠端`.gitignore`全域排除`*.pt`，且既有LFS規則只涵蓋architecture_1權重。為避免超過
GitHub一般Git單檔限制，又不漏掉使用者要求的權重，在`final/.gitattributes`新增遞迴
`*.pt` LFS規則，發布時以force-add加入被ignore的權重。34個`.pt`路徑經內容雜湊去重後
是17個LFS物件、3,613,842,852 bytes。

## 驗證方式與結果

- 最新遠端：`origin/main` commit `5feabb08815bcb4d8d6221885e76605e70f45469`，subject
  `5060ti ForVic 0827`；已在發布前fetch。
- `final/full35/verify.py`：`valid=true`，407個manifest檔案、4,493,133,705 bytes。
- `final/full35-j2-archive/verify.py`：`valid=true`，308個manifest檔案、
  2,353,874,433 bytes。
- final共34個`.pt`引用，manifest SHA統計為17個unique LFS objects；無symlink、無空檔。
- 113個`__pycache__/*.pyc`不屬manifest且受Git ignore；發布排除但不在本機刪除。
- commit前會再執行LFS attribute/pointer、staged path、diff check與兩個package verifier；
  push後會核對遠端branch hash與commit subject。
- 全程沒有啟動GPU、訓練或重新計算AP。

## 遇到的困難及解法

- 父工作樹與遠端分歧且有大量無關變更：使用最新`origin/main`的隔離worktree，避免混入。
- `.pt`同時被`.gitignore`排除且多檔超過100MB：使用final範圍的Git LFS attribute並
  明確force-add，不改寫checkpoint內容。
- `final/`含可重建Python bytecode：不刪除本機內容，只在Git提交時依ignore排除。

## 未解事項或風險

- GitHub LFS需帳號具有至少約3.37GiB可用儲存／上傳額度；若遠端拒絕，必須保留本地
  commit與全部權重，不能改成靜默漏權重的普通Git提交。
- 目前只有seed0；seed1、Partial75、FPGA/HLS及真實latency／energy尚未執行。
- 本紀錄先保存發布前證據；遠端commit hash與push結果以最終交付回報為準。
- 困難：如上；其餘無。
