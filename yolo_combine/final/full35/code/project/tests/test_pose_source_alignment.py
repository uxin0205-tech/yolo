from yolo_combine.pose_stages import pose_stage


def test_formal_pose_stages_use_full35_source_optimizer_hyperparameters() -> None:
    for name in ("p1", "p2", "p3"):
        values = pose_stage(name).trainer_overrides()
        assert values["optimizer"] == "MuSGD"
        assert values["lr0"] == 0.00038
        assert values["lrf"] == 0.5
        assert values["momentum"] == 0.948
        assert values["weight_decay"] == 0.00027
        # P1/P2/P3 use physical 128/64/32 and accumulation 1/2/4.
        # Keeping nbs=128 preserves logical exposure and the configured weight decay.
        assert values["nbs"] == 128
        assert values["cos_lr"] is True
        assert values["mosaic"] == 0.0
        assert values["close_mosaic"] == 0


def test_pose_p2_p3_document_tunable_masf_and_attention() -> None:
    assert "MASF" in pose_stage("p2").description
    assert "attention" in pose_stage("p2").description
    assert "MASF" in pose_stage("p3").description
    assert "attention" in pose_stage("p3").description
