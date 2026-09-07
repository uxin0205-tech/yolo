# 2026-09-05：qSiLU Q2 preflight schema修復

## 變更內容與原因

- V30 qSiLU Q1已完成全十區W8 PTQ並得到recover；Queue在Q2 QAT啟動前回報qSiLU q2 preflight did not pass，因此沒有開始GPU QAT。
- 依diagnosing-bugs流程建立同一ensure_q2_preflight呼叫的CPU最小重現，連續兩次約3秒內穩定失敗。
- 真實preflight內容其實是ready=true、blockers空；plan SHA位於resolved.plan_sha256。原handoff程式錯讀頂層plan_sha256，因此把健康preflight誤判為失敗。
- 新增_preflight_plan_sha256單一schema accessor；首次生成與既有artifact重用都改讀resolved.plan_sha256，沒有放寬graph、data或hash gate。
- 新增regression test，同時覆蓋首次生成與第二次重用。

## 驗證方式與結果

- Red：test_q2_preflight_uses_resolved_plan_hash_and_reuses_artifact修正前1 failed，錯誤與Queue完全相同。
- Schema probe：ready=true、blockers=[]、top-level SHA為null；resolved SHA與runtime SHA皆為25f8f50a4a0e83b14716ecf48d0e6411d89b2dc171cdb6bb1b7b991a8ad0035c。
- Green：單一regression test 1 passed in 0.67s。
- 原始真實repro連續生成及重用均成功，回傳True與相同plan SHA。
- 全專案CPU測試273 passed in 52.23s；Ruff唯一import建議已機械修正，之後Ruff、compileall與git diff --check全數通過。
- 實際q2-preflight.json已產生，SHA-256為235ad91688201bb26ccb0385a6af2cae24e7b9289489aec563535a67b70b85d7，ready=true且blockers空。
- Q1 report SHA-256為50d7449ef7728824ae23cf2b535aa5477860086062fafb6f50f424c99d79617e；dual report SHA-256為efef87f1dcc74ff23f08c19c5f812108ad32783d2fa5368625536f6e657b3928。
- Queue已由原Q1產物續跑，supervisor session為5272；第一個新事件是q2_arm_started、arm=sham、attempt=0、resume=null，證明已越過原preflight錯誤且沒有重跑Q1。

## Q1實際結果

- Dual decision為recover。
- worst total mAP50 delta為−0.020297866821，尚未通過−0.015最終門檻。
- worst total mAP50-95 delta為−0.021367118704，通過−0.04門檻。
- W8相對qSiLU matched parent的worst incremental mAP50 delta為−0.008721779556，通過−0.01 W8 incremental gate。
- 因此進入paired QAT是依契約的正確動作，目標是恢復約0.0053以上的COCO box mAP50，而不是放寬門檻。

## 假設排除

1. 已證實：report schema accessor層級錯誤。
2. 已排除：Q2 plan hash漂移；兩個真實SHA相同。
3. 已排除：殘留preflight artifact；失敗時輸出檔不存在。
4. 已排除：graph/data blocker；blockers為空。

## 困難與解法

- 困難：第一次schema probe漏設PYTHONPATH，得到鄰近但不同的ModuleNotFoundError。
- 解法：沒有把它誤當原bug；立即以原命令補上PYTHONPATH重跑，取得正確schema證據。
- 困難：apply_patch仍受既有bwrap loopback錯誤影響。
- 解法：沿用唯一sentinel且限定workspace的精確替換；沒有debug instrumentation或暫存source留下。
- 診斷用兩個/tmp JSON已刪除。

## 未解事項或風險

- Q2 paired all-W8 QAT尚未完成；preflight通過不代表最終精度會通過雙門檻。
- 若QAT沒有best_joint，V30會fail closed，不建立假的locked parent。
- formal、長epoch、multi-seed、Hardswish條件旁支與硬體量測仍未啟動。
