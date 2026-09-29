from masf_yolo.retest import queue


def test_queue_has_single_gpu_and_locked_order():
    assert queue.VARIANTS == ("PaperFormula-Full", "Lite-35", "Lite-35-F7", "Partial50-35", "Partial25-35")
    assert queue.PYTHON.name == "python"
