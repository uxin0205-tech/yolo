# 2026-09-04：整合量化流程與Paper-TWN逐區路線

## 變更內容與原因

- 新增`configs/experiments/full35-integrated-quantization-roadmap-v1.yaml`，只負責把既有v5、目前V19 W8 QAT、mixed-bit、Fixed／LS-SD4、ternary、activation coupling與formal串成同一流程；沒有修改或覆寫舊計畫與結果。
- 新增`configs/experiments/v28-paper-twn-progressive-region-plan-v1.yaml`，把三元替換改為backbone→neck→head的階段鎖定。每次只換單path，green才累加；失敗回到上一個green policy。
- 新增`docs/reports/2026-09-04-integrated-quantization-and-paper-twn-plan.md`，整理已完成結果、現況、候選區域、每階段指標、QAT條件與GPU cell上限。
- 將舊`safe`／`balanced` Paper-TWN route明確降格為失敗證據：相同paths的exact W4均green，證明不能只依weight NRMSE選三元route。
- 原始論文查核後修正命名：現有實作是`TWN-v2-0.70-layerwise-static-proxy`，不是完整Paper重現。新矩陣另要求TWN-v3 0.75 filter-wise、exact-scaled ternary、faithful filter-wise TWN QAT、TTQ及可選INQ-style progressive control分開命名與歸因。
- 既有主線保留為三條lane：W8／mixed-bit、Fixed-SD4／LS-SD4、ternary。三條只在同一locked parent與metric contract下形成Pareto後才做activation coupling。
- V19 QAT未中止；新GPU ternary工作明確等待V19完成、selector與checkpoint hash查核及parent鎖定。

## 驗證方式與結果

- 使用PyYAML解析兩份新增YAML：兩者`schema_version=1`，status與必要欄位可正常讀取。
- 將V28中21個明列graph paths與poly_shift 148-layer deployment Paper-TWN profile交叉比對：21個unique paths，missing為空。
- 逐項核對既有V24 dual regate：
  - 3-path static-safe Paper-TWN為`-0.064370／-0.111516`、reject；相同paths W4為`-0.008532／-0.012394`、green。
  - 7-path balanced Paper-TWN為`-0.037325／-0.074978`、recover；相同paths W4為`-0.008532／-0.017466`、green。
- 交叉解析Fixed-SD4 36-layer route與Paper-TWN static profile，確認新backbone／neck micro order確實分成「SD4-supported母集合內排序」與獨立Paper-distribution sentinel，沒有把兩種selection rule混稱。
- 唯讀確認V19 queue status為`qat_started`，PID仍存活；本次沒有發送signal、沒有啟動新GPU job、沒有跑formal validation。

## 困難與解法

- 困難：舊manifest的`safe`名稱容易被誤讀成mAP安全，但它只通過static NRMSE／cosine門檻。
- 解法：以同paths W4 control與完整逐任務delta重新判讀，並在新計畫中禁止重用舊3-path／7-path route。
- 困難：原始TWN不同arXiv版本的threshold不同，且論文不是純PTQ；專案現有layer-wise static用法又和逐filter演算法不同。
- 解法：釘選版本、threshold、粒度與是否訓練；歷史結果不改名覆寫，只在新計畫中加明確alias與faithfulness欄位。
- 困難：執行環境的受限sandbox一度無法建立loopback namespace，直接filesystem patch helper失敗。
- 解法：使用已核准的專用`apply_patch`行程完成同一範圍修改；未使用一般shell覆寫或超出工作目錄。

## 未解事項或風險

- V19 W8 QAT仍在執行，尚無完成manifest或最終winner；後續ternary parent目前故意不綁死。
- exact-scaled ternary與TWN-v3 filter-wise的148-layer CPU artifact尚未完成，V28維持`execution_authorized=false`。
- Paper-TWN靜態weight指標對task mAP的排序已被既有Pose route反例推翻；任何backbone／neck候選都必須完整search validation，不能由本報告直接宣稱可用。
- 逐path三元化的容量收益可能太小；final promotion除精度外還必須有實質packed-byte收益。
- 沒有native ternary kernel與target hardware量測，暫時不能宣稱延遲、功耗或吞吐改善。
- BBAT5 assignment、影像與標註均未改；無資料問題。
