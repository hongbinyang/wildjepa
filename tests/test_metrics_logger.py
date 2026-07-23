from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from wildjepa.train.common import MetricsLogger


def test_metrics_logger_writes_readable_scalars(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log_scalar("train/loss_step", 1.5, step=1)
    logger.log_scalar("train/loss_step", 1.2, step=2)
    logger.log_scalar("test/macro_f1", 0.33, step=1)
    logger.close()

    ea = EventAccumulator(str(tmp_path / "tensorboard"))
    ea.Reload()

    loss_events = ea.Scalars("train/loss_step")
    assert [e.step for e in loss_events] == [1, 2]
    assert [round(e.value, 2) for e in loss_events] == [1.5, 1.2]
    assert round(ea.Scalars("test/macro_f1")[0].value, 2) == 0.33


def test_metrics_logger_log_per_class_creates_one_tag_per_class(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log_per_class("test/per_class_f1", {0: 0.9, 5: 0.1, 12: 0.5}, step=1)
    logger.close()

    ea = EventAccumulator(str(tmp_path / "tensorboard"))
    ea.Reload()
    tags = ea.Tags()["scalars"]

    assert "test/per_class_f1/class_0" in tags
    assert "test/per_class_f1/class_5" in tags
    assert "test/per_class_f1/class_12" in tags
    assert round(ea.Scalars("test/per_class_f1/class_5")[0].value, 2) == 0.1
