import logging
import os
import sys
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from TTS.utils.generic_utils import ConsoleFormatter, setup_logger


@dataclass
class TrainTTSArgs(TrainerArgs):
    config_path: str | None = field(default=None, metadata={"help": "Path to the config file."})


def main(arg_list: list[str] | None = None):
    """Run `tts` model training directly by a `config.json` file."""
    setup_logger("TTS", level=logging.INFO, stream=sys.stdout, formatter=ConsoleFormatter())

    # init trainer args
    train_args = TrainTTSArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from command-line args
    args, config_overrides = parser.parse_known_args(arg_list)
    train_args.parse_args(args)

    # load config.json
    if args.config_path:
        # init from a file
        config = load_config(args.config_path)
        if len(config_overrides) > 0:
            config.parse_known_args(config_overrides, relaxed_parser=True)
    elif args.continue_path:
        # continue from a prev experiment
        config = load_config(os.path.join(args.continue_path, "config.json"))
        if len(config_overrides) > 0:
            config.parse_known_args(config_overrides, relaxed_parser=True)
    else:
        msg = "You need to specify either --config_path or --continue_path"
        raise RuntimeError(msg)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the model from config
    model = setup_model(config, train_samples + eval_samples)

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()
    sys.exit(0)


if __name__ == "__main__":
    main()
