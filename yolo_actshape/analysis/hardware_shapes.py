"""純算術形狀／tile／儲存分析，無 torch、checkpoint 或 GPU 依賴。"""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = {
    'reference_fixed640': (640, 640),
    'current_rect384_768': (384, 768),
    'current_rect512_1024': (512, 1024),
    'measured_square512': (512, 512),
    'proposal_rect256_512': (256, 512),
    'proposal_rect384_512': (384, 512),
    'proposal_rect512_768': (512, 768),
}


def describe(name, shape):
    h, w = shape
    assert h % 32 == w % 32 == 0 and max(shape) <= 1024
    levels = [[h // s, w // s] for s in (2, 4, 8, 16, 32)]
    # Actual selected model: head inputs P3/P4/P5 have C=256/512/512.
    head_maps = [dict(height=a, width=b, channels=c) for (a, b), c in zip(levels[2:], (256, 512, 512))]
    elements = sum(m['height'] * m['width'] * m['channels'] for m in head_maps)
    n = math.prod(levels[-1])
    # This is a full materialized score matrix, not the minimum SRAM requirement.
    score_values_per_site = 4 * n * n
    tiles = {}
    for t in (4, 8, 16, 32):
        tiles[str(t)] = []
        for a, b in levels:
            capacity = math.ceil(a / t) * math.ceil(b / t) * t * t
            tiles[str(t)].append({'valid_cells': a*b, 'capacity': capacity,
                                  'utilization': a*b/capacity,
                                  'allocation_overhead': capacity/(a*b)-1})
    return {'name': name, 'input_hw': list(shape), 'levels_P1_P5': levels,
            'head_maps': head_maps, 'all_axes_power_of_two': all(x & (x-1) == 0 for hw in levels for x in hw),
            'input_area_ratio_fixed640': h*w/(640*640),
            'attention_tokens': n, 'attention_pair_ratio_N400': (n/400)**2,
            'attention_query_tiles64': math.ceil(n/64),
            'attention_query_tile64_utilization': n/(math.ceil(n/64)*64),
            'denominator_max_code': n*32768, 'denominator_unsigned_bits': (n*32768).bit_length(),
            'three_head_maps_sum_FP32_bytes': 4*elements,
            'three_head_maps_sum_if_INT8_bytes': elements,
            'per_site_full_score_FP32_bytes': 4*score_values_per_site,
            'spatial_tiles': tiles,
            'estimate_only': True}


def main():
    rows = [describe(n, hw) for n, hw in SHAPES.items()]
    assert describe('test', (512, 768))['spatial_tiles']['8'][-1]['utilization'] == 1
    assert math.ceil(40/8)**2*8**2 == 1600
    assert math.ceil(40/16)**2*16**2 == 2304
    output = {'schema': 'hardware-shape-estimates-v1', 'GPU_used': False, 'rows': rows,
              'assumptions': ['既有五次stride2與head C=256/512/512，batch1；僅幾何推導，新增尺寸未驗證任務AP。',
                  'spatial tile容量不含卷積halo、通道padding、雙緩衝、權重、skip liveness與DMA。',
                  '三個head輸入的單份容量總和不是實際peak SRAM；INT8只是未實作的位寬情境。',
                  'score矩陣是完整物化成本，可經分塊／重算降低；不代表必須同時放入SRAM。',
                  '實際延遲／資源取決於layout、parallel lanes、bank、ports、bus及目標裝置。',
                  '不改變目前GPU queue、權重、qSiLU、D1、B2或S1。']}
    (HERE / 'hardware-shapes.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    with (HERE / 'hardware-shapes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name','input_hw','P3','P4','P5','N','area_ratio','attention_pairs_ratio','head_map_FP32_MiB','spatial8_P5_utilization'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'name':row['name'],'input_hw':'x'.join(map(str,row['input_hw'])),
                **{k:'x'.join(map(str,row['levels_P1_P5'][i])) for k,i in [('P3',2),('P4',3),('P5',4)]},
                'N':row['attention_tokens'],'area_ratio':row['input_area_ratio_fixed640'],
                'attention_pairs_ratio':row['attention_pair_ratio_N400'],
                'head_map_FP32_MiB':row['three_head_maps_sum_FP32_bytes']/2**20,
                'spatial8_P5_utilization':row['spatial_tiles']['8'][-1]['utilization']})
    print('PASS: 7組純CPU幾何／tile／儲存估算；未使用GPU或改動queue')


if __name__ == '__main__':
    main()
