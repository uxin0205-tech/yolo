# MASF-YOLO Phase 1 Report

Pipeline state: running / report
Dataset hash: 6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc
Environment: Ultralytics 8.4.90, PyTorch 2.11.0+cu128, device NVIDIA GeForce RTX 5090
BEST_PARTIAL: M2
Selection reason: quality_hardware_ranking
Final audit: PASS

## Data exposure warning

B0, B1, and every MFAM variant use a pose-derived initializer that has already seen BBT5. All metrics are data-exposed operational ablations, not leak-free generalization estimates.
Checkpoint hash: 9adacdd1a86cde27b7568c0756ca06f7be83160445fc90a10449206c82b06f4d
Provenance: BBT5 pose-trained checkpoint with the pose head removed and a detection head installed

## Training budget warning

SP2P 使用序列式訓練預算：先繼承已完成 10+90 epochs 的 SP2-B 與 BEST_PARTIAL，再執行自己的 10+90 epochs。其結果不可直接視為與僅從 B1-B 訓練 100 epochs 的單段架構公平消融。

## Training artifacts

- b1_a: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/b1_a/canonical.pt`, strict reload passed
- b1_b: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/b1_b/canonical.pt`, strict reload passed
- formal_m7: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_m7/canonical.pt`, strict reload passed
- formal_m0: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_m0/canonical.pt`, strict reload passed
- formal_m1: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_m1/canonical.pt`, strict reload passed
- formal_m2: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_m2/canonical.pt`, strict reload passed
- formal_m3: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_m3/canonical.pt`, strict reload passed
- formal_p3m: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/formal_p3m/canonical.pt`, strict reload passed
- sp2_a: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/sp2_a/canonical.pt`, strict reload passed
- sp2_b: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/sp2_b/canonical.pt`, strict reload passed
- sp2p_a: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/sp2p_a/canonical.pt`, strict reload passed
  - architecture=SP2M2, BEST_PARTIAL=M2, parents={'formal_m2': '95f0bcc3555384715cc43ee11b8cde84f08aa7c2ee8044111c69562e6b645d38', 'sp2_b': '089f21355ebcd85240d77ef6d9a3c90bc15eec0587cbeffb62260699873b03d5'}
- sp2p_b: canonical `/home/uxin/yolo/yolo_masf/artifacts/static-phase1/training/sp2p_b/canonical.pt`, strict reload passed
  - architecture=SP2M2, BEST_PARTIAL=M2, parents={'formal_m2': '95f0bcc3555384715cc43ee11b8cde84f08aa7c2ee8044111c69562e6b645d38', 'sp2_b': '089f21355ebcd85240d77ef6d9a3c90bc15eec0587cbeffb62260699873b03d5'}

## Evaluation and profiling

- B0: val mAP50-95=0.7508155882154294, test mAP50-95=0.7708117389613107, GFLOPs=67.6461056

### B0

- Validation overall: mAP50-95=0.7508
  - Ball: AP50=0.9825, AP50-95=0.6325, precision=0.4146, recall=0.9903, GT=414, predictions=989, missed=4, false positives=579
    - size/blur recall: tiny=1.0000 (n=19), small=0.9962 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9978, AP50-95=0.8691, precision=0.6657, recall=1.0000, GT=685, predictions=1029, missed=0, false positives=344
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=1.0000 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7708
  - Ball: AP50=0.8876, AP50-95=0.6784, precision=0.2876, recall=0.9662, GT=503, predictions=1690, missed=17, false positives=1204
    - size/blur recall: tiny=0.6400 (n=25), small=0.9570 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9891, AP50-95=0.8632, precision=0.5480, recall=0.9966, GT=590, predictions=1073, missed=2, false positives=485
    - size/blur recall: tiny=null (n=0), small=0.9848 (n=66), large=0.9981 (n=524), blur=0.9961 (n=256)
- B1: val mAP50-95=0.7081035800839978, test mAP50-95=0.7303279443334882, GFLOPs=87.4307072

### B1

- Validation overall: mAP50-95=0.7081
  - Ball: AP50=0.9477, AP50-95=0.5648, precision=0.4217, recall=0.9758, GT=414, predictions=958, missed=10, false positives=554
    - size/blur recall: tiny=0.8421 (n=19), small=0.9848 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9878, AP50-95=0.8514, precision=0.3309, recall=0.9985, GT=685, predictions=2067, missed=1, false positives=1383
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7303
  - Ball: AP50=0.8281, AP50-95=0.6126, precision=0.2338, recall=0.9702, GT=503, predictions=2087, missed=15, false positives=1599
    - size/blur recall: tiny=0.6400 (n=25), small=0.9677 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9846, AP50-95=0.8481, precision=0.3575, recall=0.9932, GT=590, predictions=1639, missed=4, false positives=1053
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9924 (n=524), blur=1.0000 (n=256)
- M7: val mAP50-95=0.7078918875723638, test mAP50-95=0.738556489883728, GFLOPs=88.5841408

### M7

- Validation overall: mAP50-95=0.7079
  - Ball: AP50=0.9599, AP50-95=0.5717, precision=0.3140, recall=0.9831, GT=414, predictions=1296, missed=7, false positives=889
    - size/blur recall: tiny=0.8947 (n=19), small=0.9924 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9884, AP50-95=0.8440, precision=0.3363, recall=0.9985, GT=685, predictions=2034, missed=1, false positives=1350
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7386
  - Ball: AP50=0.8847, AP50-95=0.6324, precision=0.1399, recall=0.9702, GT=503, predictions=3488, missed=15, false positives=3000
    - size/blur recall: tiny=0.6400 (n=25), small=0.9677 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9864, AP50-95=0.8447, precision=0.2476, recall=0.9932, GT=590, predictions=2367, missed=4, false positives=1781
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9924 (n=524), blur=1.0000 (n=256)
- M0: val mAP50-95=0.7107597336511077, test mAP50-95=0.7234334402365616, GFLOPs=88.7021056

### M0

- Validation overall: mAP50-95=0.7108
  - Ball: AP50=0.9538, AP50-95=0.5750, precision=0.3392, recall=0.9831, GT=414, predictions=1200, missed=7, false positives=793
    - size/blur recall: tiny=0.8421 (n=19), small=0.9962 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9887, AP50-95=0.8465, precision=0.3631, recall=0.9985, GT=685, predictions=1884, missed=1, false positives=1200
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7234
  - Ball: AP50=0.8373, AP50-95=0.6113, precision=0.1762, recall=0.9702, GT=503, predictions=2769, missed=15, false positives=2281
    - size/blur recall: tiny=0.6400 (n=25), small=0.9677 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9832, AP50-95=0.8356, precision=0.3209, recall=0.9932, GT=590, predictions=1826, missed=4, false positives=1240
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9924 (n=524), blur=1.0000 (n=256)
- M1: val mAP50-95=0.7095032534045789, test mAP50-95=0.7263363732546134, GFLOPs=88.4923904

### M1

- Validation overall: mAP50-95=0.7095
  - Ball: AP50=0.9498, AP50-95=0.5704, precision=0.3169, recall=0.9807, GT=414, predictions=1281, missed=8, false positives=875
    - size/blur recall: tiny=0.8421 (n=19), small=0.9924 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9884, AP50-95=0.8486, precision=0.3528, recall=0.9985, GT=685, predictions=1939, missed=1, false positives=1255
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7263
  - Ball: AP50=0.8301, AP50-95=0.6162, precision=0.1784, recall=0.9682, GT=503, predictions=2730, missed=16, false positives=2243
    - size/blur recall: tiny=0.6400 (n=25), small=0.9677 (n=186), large=0.9966 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9830, AP50-95=0.8365, precision=0.3401, recall=0.9932, GT=590, predictions=1723, missed=4, false positives=1137
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9924 (n=524), blur=1.0000 (n=256)
- M2: val mAP50-95=0.7047367379755087, test mAP50-95=0.7250490833578963, GFLOPs=87.7518336

### M2

- Validation overall: mAP50-95=0.7047
  - Ball: AP50=0.9481, AP50-95=0.5613, precision=0.3923, recall=0.9807, GT=414, predictions=1035, missed=8, false positives=629
    - size/blur recall: tiny=1.0000 (n=19), small=0.9848 (n=264), large=0.9695 (n=131), blur=null (n=0)
  - Bat: AP50=0.9881, AP50-95=0.8482, precision=0.3319, recall=0.9985, GT=685, predictions=2061, missed=1, false positives=1377
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7250
  - Ball: AP50=0.8255, AP50-95=0.6089, precision=0.1985, recall=0.9702, GT=503, predictions=2459, missed=15, false positives=1971
    - size/blur recall: tiny=0.6400 (n=25), small=0.9677 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9831, AP50-95=0.8412, precision=0.3470, recall=0.9932, GT=590, predictions=1689, missed=4, false positives=1103
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9924 (n=524), blur=1.0000 (n=256)
- M3: val mAP50-95=0.7020275298079648, test mAP50-95=0.7262233791704938, GFLOPs=87.5388416

### M3

- Validation overall: mAP50-95=0.7020
  - Ball: AP50=0.9374, AP50-95=0.5539, precision=0.3562, recall=0.9783, GT=414, predictions=1137, missed=9, false positives=732
    - size/blur recall: tiny=0.8421 (n=19), small=0.9886 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9883, AP50-95=0.8502, precision=0.3539, recall=0.9985, GT=685, predictions=1933, missed=1, false positives=1249
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7262
  - Ball: AP50=0.8285, AP50-95=0.6143, precision=0.2013, recall=0.9662, GT=503, predictions=2414, missed=17, false positives=1928
    - size/blur recall: tiny=0.6400 (n=25), small=0.9570 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9781, AP50-95=0.8381, precision=0.3452, recall=0.9881, GT=590, predictions=1689, missed=7, false positives=1106
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9866 (n=524), blur=0.9883 (n=256)
- P3M: val mAP50-95=0.6990248525417414, test mAP50-95=0.7339847041762806, GFLOPs=88.4268544

### P3M

- Validation overall: mAP50-95=0.6990
  - Ball: AP50=0.9539, AP50-95=0.5535, precision=0.3853, recall=0.9783, GT=414, predictions=1051, missed=9, false positives=646
    - size/blur recall: tiny=0.8421 (n=19), small=0.9886 (n=264), large=0.9771 (n=131), blur=null (n=0)
  - Bat: AP50=0.9878, AP50-95=0.8445, precision=0.3511, recall=0.9985, GT=685, predictions=1948, missed=1, false positives=1264
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7340
  - Ball: AP50=0.8577, AP50-95=0.6248, precision=0.1716, recall=0.9662, GT=503, predictions=2832, missed=17, false positives=2346
    - size/blur recall: tiny=0.6400 (n=25), small=0.9570 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9866, AP50-95=0.8432, precision=0.3745, recall=0.9966, GT=590, predictions=1570, missed=2, false positives=982
    - size/blur recall: tiny=null (n=0), small=1.0000 (n=66), large=0.9962 (n=524), blur=1.0000 (n=256)
- SP2: val mAP50-95=0.7056195082805379, test mAP50-95=0.7170246881882252, GFLOPs=80.528128

### SP2

- Validation overall: mAP50-95=0.7056
  - Ball: AP50=0.9405, AP50-95=0.5581, precision=0.2127, recall=0.9734, GT=414, predictions=1895, missed=11, false positives=1492
    - size/blur recall: tiny=0.8421 (n=19), small=0.9848 (n=264), large=0.9695 (n=131), blur=null (n=0)
  - Bat: AP50=0.9873, AP50-95=0.8531, precision=0.3470, recall=0.9985, GT=685, predictions=1971, missed=1, false positives=1287
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7170
  - Ball: AP50=0.8161, AP50-95=0.5970, precision=0.0667, recall=0.9821, GT=503, predictions=7402, missed=9, false positives=6908
    - size/blur recall: tiny=0.6400 (n=25), small=1.0000 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9766, AP50-95=0.8370, precision=0.3712, recall=0.9864, GT=590, predictions=1568, missed=8, false positives=986
    - size/blur recall: tiny=null (n=0), small=0.9848 (n=66), large=0.9866 (n=524), blur=0.9883 (n=256)
- SP2P: val mAP50-95=0.6971090161588444, test mAP50-95=0.7212682825058977, GFLOPs=80.8492544

### SP2P

- Validation overall: mAP50-95=0.6971
  - Ball: AP50=0.9268, AP50-95=0.5472, precision=0.2462, recall=0.9662, GT=414, predictions=1625, missed=14, false positives=1225
    - size/blur recall: tiny=0.6842 (n=19), small=0.9848 (n=264), large=0.9695 (n=131), blur=null (n=0)
  - Bat: AP50=0.9872, AP50-95=0.8470, precision=0.3298, recall=0.9985, GT=685, predictions=2074, missed=1, false positives=1390
    - size/blur recall: tiny=1.0000 (n=3), small=1.0000 (n=60), large=0.9984 (n=622), blur=1.0000 (n=293)
- Test overall: mAP50-95=0.7213
  - Ball: AP50=0.8459, AP50-95=0.6047, precision=0.0661, recall=0.9821, GT=503, predictions=7477, missed=9, false positives=6983
    - size/blur recall: tiny=0.6400 (n=25), small=1.0000 (n=186), large=1.0000 (n=292), blur=1.0000 (n=13)
  - Bat: AP50=0.9817, AP50-95=0.8378, precision=0.3072, recall=0.9915, GT=590, predictions=1904, missed=5, false positives=1319
    - size/blur recall: tiny=null (n=0), small=0.9848 (n=66), large=0.9924 (n=524), blur=0.9961 (n=256)

## Audit errors

- None
