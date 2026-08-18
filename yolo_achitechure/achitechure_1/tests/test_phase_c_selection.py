from achitechure_1.selection import phase_c_candidate


def test_phase_c_retains_parent_until_child_strictly_improves() -> None:
    assert phase_c_candidate("parent", 0.5, "child", 0.5) == "parent"
    assert phase_c_candidate("parent", 0.5, "child", 0.50001) == "child"
