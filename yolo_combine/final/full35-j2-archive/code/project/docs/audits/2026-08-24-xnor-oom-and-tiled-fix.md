# Full35 XNOR OOM 根因與 exact tiled 修正

## 原始證據

未截斷 session：

```text
/home/uxin/.codex/sessions/2026/08/12/
rollout-2026-08-12T23-26-55-019ff695-6c8b-7db0-971b-38c1cd54637f.jsonl
```

line 3624 顯示 epoch2 train 380/380完成後，第二次 validation完成12/18，在第13 batch
進入 `attention.py → binary_basis.py:xnor_popcount_dot` 時嘗試配置5.94GiB失敗。

train physical batch16被 validator自動加倍成32；rect batch第13組實際672×672，P5 token
為21×21=441。boolean equality後的預設 int64 reduction workspace形狀為
`[32,4,441,441,32]`：

```text
32 × 4 × 441 × 441 × 32 × 8 bytes = 5.935089 GiB
```

當時外部程序佔用17.68GiB、free僅2.90GiB，使錯誤更容易發生；但核心的
`O(B·H·N²·D·8)` allocation本身確實存在。

## physical128 影響

- train B128、640、N400：untiled單一 workspace約19.531GiB。
- 若 validator沿用2×train，val B256、rect672、N441：約47.481GiB，單一 allocation
  已大於 RTX5090總VRAM。

因此不能只調 allocator或假設空卡即可。

## 修正

`src/yolo_combine/xnor.py` 在 query與key token各以32分塊，但一次完成完整channel
integer reduction。這保留 forward逐值結果與現有 boolean Q/K gradient semantics。

- train B128 tile32 workspace上界：0.125GiB。
- val B256 tile32 workspace上界：0.25GiB。

測試包含 literal oracle、broadcast逐值比較、gradient semantics與 workspace計算。這只
修正主導中間張量；完整 B128 train仍需空卡 profile，不能由靜態估算宣稱已可行。

