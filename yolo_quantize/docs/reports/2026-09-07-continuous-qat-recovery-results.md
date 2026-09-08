# 同 parent 短 QAT 恢復結果（持續更新）

已完成 6/6 組；僅搜尋驗證，非 formal 或重新 export 驗證。

所有下降為相對 accepted 模型的絕對百分點，包含 activation 替換；16 項全過才達標。
PTQ 未達標不代表 QAT 無效；回升幅度須逐項比，不能相減不同指標的最差值當作恢復量。

| 實驗 | 回合 | 最差 mAP50 下降（pp） | 最差 mAP50–95 下降（pp） | 搜尋雙門檻 |
| --- | --- | ---: | ---: | --- |
| pose-sd4-fixed | 1 | 1.043 | 2.142 | 通過 |
| pose-sd4-fixed | 2 | 0.934 | 1.124 | 通過 |
| pose-sd4-fixed | 3 | 0.997 | 1.103 | 通過 |
| pose-sd4-fixed | 4 | 0.849 | 1.041 | 通過 |
| pose-sd4-fixed | 5 | 0.834 | 1.008 | 通過 |
| pose-ls-sd4 | 1 | 1.043 | 2.142 | 通過 |
| pose-ls-sd4 | 2 | 0.968 | 1.092 | 通過 |
| pose-ls-sd4 | 3 | 0.907 | 1.116 | 通過 |
| pose-ls-sd4 | 4 | 0.764 | 0.975 | 通過 |
| pose-ls-sd4 | 5 | 0.896 | 1.043 | 通過 |
| masf-exact-ternary | 1 | 0.997 | 1.257 | 通過 |
| masf-exact-ternary | 2 | 0.900 | 1.186 | 通過 |
| masf-exact-ternary | 3 | 1.007 | 1.172 | 通過 |
| masf-exact-ternary | 4 | 0.829 | 1.007 | 通過 |
| masf-exact-ternary | 5 | 0.949 | 1.069 | 通過 |
| masf-paper-twn | 1 | 1.023 | 1.305 | 通過 |
| masf-paper-twn | 2 | 0.903 | 1.146 | 通過 |
| masf-paper-twn | 3 | 0.984 | 1.159 | 通過 |
| masf-paper-twn | 4 | 0.788 | 1.043 | 通過 |
| masf-paper-twn | 5 | 0.874 | 1.064 | 通過 |
| masf-twn | 1 | 0.996 | 1.229 | 通過 |
| masf-twn | 2 | 0.915 | 1.188 | 通過 |
| masf-twn | 3 | 1.010 | 1.171 | 通過 |
| masf-twn | 4 | 0.872 | 1.023 | 通過 |
| masf-twn | 5 | 0.826 | 0.981 | 通過 |
| detect-predictor-ls-sd4-recovery | 1 | 1.823 | 5.682 | 未通過 |
| detect-predictor-ls-sd4-recovery | 2 | 1.118 | 1.289 | 通過 |
| detect-predictor-ls-sd4-recovery | 3 | 1.082 | 1.198 | 通過 |
| detect-predictor-ls-sd4-recovery | 4 | 1.072 | 1.190 | 通過 |
| detect-predictor-ls-sd4-recovery | 5 | 1.016 | 1.147 | 通過 |

## pose-sd4-fixed：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 96.517 | 96.611 | +0.094 |
| bbat/ball/box/map50_95 | 75.496 | 77.853 | +2.357 |
| bbat/ball/pose/map50 | 96.548 | 96.626 | +0.078 |
| bbat/ball/pose/map50_95 | 96.515 | 96.608 | +0.093 |
| bbat/bat/box/map50 | 99.177 | 99.263 | +0.086 |
| bbat/bat/box/map50_95 | 87.284 | 88.853 | +1.569 |
| bbat/bat/pose/map50 | 99.191 | 99.265 | +0.074 |
| bbat/bat/pose/map50_95 | 99.160 | 99.236 | +0.076 |
| bbat/box/map50 | 97.847 | 97.937 | +0.090 |
| bbat/box/map50_95 | 81.390 | 83.353 | +1.963 |
| bbat/pose/map50 | 97.870 | 97.946 | +0.076 |
| bbat/pose/map50_95 | 97.837 | 97.922 | +0.085 |
| coco/box/map50 | 66.237 | 66.238 | +0.001 |
| coco/box/map50_95 | 48.646 | 48.794 | +0.149 |
| coco/person/box/map50 | 83.441 | 83.364 | -0.076 |
| coco/person/box/map50_95 | 61.173 | 61.102 | -0.072 |


## pose-ls-sd4：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 96.517 | 96.439 | -0.078 |
| bbat/ball/box/map50_95 | 75.496 | 78.096 | +2.600 |
| bbat/ball/pose/map50 | 96.548 | 96.525 | -0.024 |
| bbat/ball/pose/map50_95 | 96.515 | 96.511 | -0.004 |
| bbat/bat/box/map50 | 99.177 | 99.306 | +0.129 |
| bbat/bat/box/map50_95 | 87.284 | 88.788 | +1.505 |
| bbat/bat/pose/map50 | 99.191 | 99.306 | +0.115 |
| bbat/bat/pose/map50_95 | 99.160 | 99.283 | +0.123 |
| bbat/box/map50 | 97.847 | 97.873 | +0.026 |
| bbat/box/map50_95 | 81.390 | 83.442 | +2.052 |
| bbat/pose/map50 | 97.870 | 97.915 | +0.046 |
| bbat/pose/map50_95 | 97.837 | 97.897 | +0.060 |
| coco/box/map50 | 66.237 | 66.176 | -0.061 |
| coco/box/map50_95 | 48.646 | 48.759 | +0.114 |
| coco/person/box/map50 | 83.441 | 83.469 | +0.029 |
| coco/person/box/map50_95 | 61.173 | 61.163 | -0.010 |


## masf-exact-ternary：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 97.194 | 96.972 | -0.222 |
| bbat/ball/box/map50_95 | 77.716 | 77.626 | -0.089 |
| bbat/ball/pose/map50 | 97.194 | 96.992 | -0.202 |
| bbat/ball/pose/map50_95 | 97.175 | 96.992 | -0.183 |
| bbat/bat/box/map50 | 99.192 | 99.355 | +0.163 |
| bbat/bat/box/map50_95 | 88.870 | 88.920 | +0.050 |
| bbat/bat/pose/map50 | 99.212 | 99.369 | +0.156 |
| bbat/bat/pose/map50_95 | 99.186 | 99.341 | +0.154 |
| bbat/box/map50 | 98.193 | 98.163 | -0.030 |
| bbat/box/map50_95 | 83.293 | 83.273 | -0.020 |
| bbat/pose/map50 | 98.203 | 98.181 | -0.023 |
| bbat/pose/map50_95 | 98.181 | 98.166 | -0.015 |
| coco/box/map50 | 66.152 | 66.123 | -0.029 |
| coco/box/map50_95 | 48.586 | 48.733 | +0.147 |
| coco/person/box/map50 | 83.316 | 83.448 | +0.132 |
| coco/person/box/map50_95 | 61.037 | 61.180 | +0.144 |


## masf-paper-twn：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 97.155 | 96.932 | -0.223 |
| bbat/ball/box/map50_95 | 77.251 | 77.695 | +0.444 |
| bbat/ball/pose/map50 | 97.155 | 96.943 | -0.212 |
| bbat/ball/pose/map50_95 | 97.132 | 96.934 | -0.197 |
| bbat/bat/box/map50 | 99.262 | 99.294 | +0.032 |
| bbat/bat/box/map50_95 | 88.680 | 88.481 | -0.199 |
| bbat/bat/pose/map50 | 99.279 | 99.307 | +0.028 |
| bbat/bat/pose/map50_95 | 99.246 | 99.282 | +0.036 |
| bbat/box/map50 | 98.209 | 98.113 | -0.096 |
| bbat/box/map50_95 | 82.965 | 83.088 | +0.123 |
| bbat/pose/map50 | 98.217 | 98.125 | -0.092 |
| bbat/pose/map50_95 | 98.189 | 98.108 | -0.081 |
| coco/box/map50 | 66.084 | 66.197 | +0.113 |
| coco/box/map50_95 | 48.532 | 48.738 | +0.206 |
| coco/person/box/map50 | 83.436 | 83.365 | -0.071 |
| coco/person/box/map50_95 | 61.133 | 61.168 | +0.036 |


## masf-twn：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 97.148 | 97.031 | -0.117 |
| bbat/ball/box/map50_95 | 77.628 | 77.951 | +0.323 |
| bbat/ball/pose/map50 | 97.148 | 97.062 | -0.086 |
| bbat/ball/pose/map50_95 | 97.127 | 97.046 | -0.081 |
| bbat/bat/box/map50 | 99.205 | 99.288 | +0.083 |
| bbat/bat/box/map50_95 | 88.702 | 88.642 | -0.060 |
| bbat/bat/pose/map50 | 99.237 | 99.307 | +0.070 |
| bbat/bat/pose/map50_95 | 99.204 | 99.284 | +0.081 |
| bbat/box/map50 | 98.176 | 98.160 | -0.017 |
| bbat/box/map50_95 | 83.165 | 83.297 | +0.131 |
| bbat/pose/map50 | 98.192 | 98.184 | -0.008 |
| bbat/pose/map50_95 | 98.165 | 98.165 | -0.000 |
| coco/box/map50 | 66.156 | 66.245 | +0.090 |
| coco/box/map50_95 | 48.595 | 48.821 | +0.226 |
| coco/person/box/map50 | 83.374 | 83.404 | +0.030 |
| coco/person/box/map50_95 | 61.168 | 61.149 | -0.019 |


## detect-predictor-ls-sd4-recovery：第 5 回合對 PTQ

| 指標 | PTQ（%） | QAT（%） | QAT − PTQ（pp） |
| --- | ---: | ---: | ---: |
| bbat/ball/box/map50 | 97.165 | 97.060 | -0.105 |
| bbat/ball/box/map50_95 | 77.252 | 78.281 | +1.028 |
| bbat/ball/pose/map50 | 97.165 | 97.060 | -0.105 |
| bbat/ball/pose/map50_95 | 97.149 | 97.054 | -0.095 |
| bbat/bat/box/map50 | 99.230 | 99.305 | +0.075 |
| bbat/bat/box/map50_95 | 88.831 | 88.728 | -0.103 |
| bbat/bat/pose/map50 | 99.248 | 99.316 | +0.069 |
| bbat/bat/pose/map50_95 | 99.217 | 99.287 | +0.070 |
| bbat/box/map50 | 98.198 | 98.183 | -0.015 |
| bbat/box/map50_95 | 83.042 | 83.505 | +0.463 |
| bbat/pose/map50 | 98.206 | 98.188 | -0.018 |
| bbat/pose/map50_95 | 98.183 | 98.170 | -0.012 |
| coco/box/map50 | 65.497 | 66.056 | +0.559 |
| coco/box/map50_95 | 44.464 | 48.655 | +4.192 |
| coco/person/box/map50 | 83.161 | 83.497 | +0.336 |
| coco/person/box/map50_95 | 56.800 | 61.261 | +4.461 |


完整五回合逐項數據及來源 SHA-256：`artifacts/queues/full-model-continuous-0907/qat-recovery-summary.json`。

外部 sham 僅歷史參考，不宣稱同 parent 的 sham 因果對照。六組完成後才選累積混合配置；不得直接拼接各區獨立最優。
