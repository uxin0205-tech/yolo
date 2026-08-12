from masf_yolo.retest.profile_all import profile_all


def test_profile_module_entrypoint_exists():
    assert callable(profile_all)
