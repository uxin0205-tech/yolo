import json

from binaryqk.report import plot


def test_plot_uses_small_json_without_checkpoint(tmp_path):
    payload = {'curves': [{'variant': 'B2', 'phase': phase, 'epochs': [0, 1],
                          'official_AP50_95': [.4, .45]} for phase in ('screening', 'recovery')]}
    (tmp_path / 'comparison.json').write_text(json.dumps(payload))
    plot(tmp_path)
    assert '<svg' in (tmp_path / 'curves.svg').read_text()
    assert not list(tmp_path.glob('*.pt'))
