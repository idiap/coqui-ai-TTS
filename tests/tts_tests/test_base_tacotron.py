from types import SimpleNamespace

from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2


def _make_trainer(config, *, train_samples, eval_samples):
    return SimpleNamespace(
        config=config,
        total_steps_done=0,
        train_samples=train_samples,
        eval_samples=eval_samples,
        train_loader="train_loader",
        eval_loader="eval_loader",
    )


def test_gradual_training_resets_loaders_on_schedule_change():
    config = Tacotron2Config(num_chars=32, r=2, batch_size=2, gradual_training=[[0, 2, 2], [1, 1, 2]])
    model = Tacotron2(config)
    trainer = _make_trainer(config, train_samples=["sample"], eval_samples=["sample"])

    # schedule unchanged: cached loaders are kept
    model.on_epoch_start(trainer)
    assert trainer.train_loader == "train_loader"
    assert trainer.eval_loader == "eval_loader"
    assert model.decoder.r == 2

    # schedule changes `r`: loaders are dropped so the trainer rebuilds them
    trainer.total_steps_done = 1
    model.on_epoch_start(trainer)
    assert trainer.train_loader is None
    assert trainer.eval_loader is None
    assert config.r == 1
    assert config.batch_size == 2
    assert model.decoder.r == 1


def test_gradual_training_keeps_user_loaders():
    config = Tacotron2Config(num_chars=32, r=2, batch_size=2, gradual_training=[[0, 2, 2], [1, 1, 4]])
    model = Tacotron2(config)
    trainer = _make_trainer(config, train_samples=None, eval_samples=None)

    trainer.total_steps_done = 1
    model.on_epoch_start(trainer)
    assert trainer.train_loader == "train_loader"
    assert trainer.eval_loader == "eval_loader"
    assert config.r == 1
    assert config.batch_size == 4
