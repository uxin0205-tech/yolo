# 第四階段延長與 LR 方法比較

每個任務從同一 E5 resume 獨立分成 L1、L2；保留 live／EMA／optimizer／scaler。先完整評估已訓 E5 alpha-on/off，再四個 smoke，Pose 短週期研究先行，接著 Detect。每組增加 5～15 輪，以 primary AP 的 5 輪 patience 早停；最多累計 20 輪。完成後只整理第四階段，不啟動 HOG。


2026-10-20 09:50 已完成並停止，15／15 jobs 成功。[完整結果](RESULTS.md) 與 [交付稽核](final-audit.json) 已整理，兩任務均建議保留原 parent，第五階段待指定。
