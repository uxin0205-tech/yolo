# Study data

正式資料是 `bbt5-detect-baseline/dataset`，不是 COCO，也不是其他 sibling repo
的資料。固定輸出 split 位於 `artifacts/static-phase1/dataset/`：train 5093、val
781、test 773，類別順序為 `ball=0, bat=1`，split ratio 為 80/10/10。

資料 manifest：`artifacts/static-phase1/dataset/manifest.json`。
其 dataset hash 為 `6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc`。
原始資料唯讀保留，不由後處理搬移或覆寫。
