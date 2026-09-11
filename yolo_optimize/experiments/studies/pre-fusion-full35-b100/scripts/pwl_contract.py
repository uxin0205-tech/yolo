"""本研究正式 PWL 契約：[-10,0]，20 段，固定表格。"""


def verify_pwl(model):
    sites = []
    for name, module in model.named_modules():
        if type(module).__name__ not in ('PiecewiseLinearSoftmax', 'BitTruePiecewiseLinearSoftmax'):
            continue
        assert module.score_floor == -10.0 and module.segments == 20, name
        assert sum(p.numel() for p in module.parameters()) == 0, name
        sites.append({'site': name, 'class': type(module).__name__,
                      'range': [-10., 0.], 'segments': 20, 'segment_width': .5})
    assert len(sites) == 2, sites
    return sites
