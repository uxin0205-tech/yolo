# result figures

本目錄放置由正式結果產生、可直接嵌入 Markdown 的 SVG 圖表。

- 精度來源：`../comparison.csv`。
- 計算量來源：修正後的 `write_variant_profile()`，固定兩個 Attention sites、每處 400 tokens、4 heads、$d_k=32$、$d_v=64$。
- 大小來源：清理前量測的實際檔案 bytes 與 checkpoint 內 parameter/buffer tensor bytes；部分階段權重已依 `../CLEANUP.md` 刪除。
- 輸出：主線 mAP、normalization、精度－成本與 checkpoint 大小圖。
- `bdcn-v3-map.svg`：BDCN V2/V3 修正前後的 mAP50-95。
- Git：SVG 與本 README 應提交；原始 Ultralytics batch/curve 圖不提交。

圖中的 arithmetic cost 是演算法 proxy，不是 FLOPs、GPU latency、FPGA cycles 或 energy；memory traffic 也是元素存取 proxy，不是實測 bytes。
