import argparse
import json
import os
from typing import List, Dict
from typing import get_args, get_origin, Union, Any, Optional

import yaml
from pydantic import BaseModel, Field


def try_parse_dict_string(s):
    """
    Attempts to parse a string as a JSON dictionary.

    :param s: The string to parse.
    :return: If successful, returns the dictionary; otherwise, returns None.
    """
    try:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return s
        s = s.replace("'", '"')
        s = s.replace("True", "true").replace("False", "false")
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError) as e:
        print(e)
        pass
    return None


class CorpusConfig(BaseModel):
    """
    Configuration for a single corpus.
    """
    dataset: str = Field(
        ...,
        description="Name of the corpus."
    )
    split: Optional[str] = Field(
        default=None,
        description="Data split, e.g., 'train', 'validation'."
    )
    text_column: str = Field(
        "clear_text",
        description="Name of the text column."
    )
    id_column: str = Field(
        "audio_id",
        description="Name of the id column."
    )
    audio_column: str = Field(
        "audio",
        description="Name of audio column."
    )
    load_from_disk: bool = Field(
        default=False,
        description="Whether to use load_dataset or load_from_disk method"
    )
    load_dataset_kwargs: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional dataset loading kwargs."
    )
    target_duration_hours: Optional[float] = Field(
        default=None,
        description="The target duration of this split in hours."
    )

    @property
    def target_duration_seconds(self):
        return self.target_duration_hours * 3600 if self.target_duration_hours is not None else None

    def __init__(self, **data: Any):
        data["split"] = None if data.get("split", "*") == "*" else data["split"]
        super().__init__(**data)


class TrainArgs(BaseModel):
    """
    Arguments for training configuration
    """
    n_steps: int = Field(
        default=250,
        description="Number of training steps."
    )
    batch_size: int = Field(
        default=8,
        description="Batch size for training."
    )
    eval_batch_size: int = Field(
        default=1,
        description="Batch size for evaluation."
    )
    gradient_accumulation_steps: int = Field(
        default=160,
        description="Number of steps to accumulate gradients before updating the model."
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Whether to use gradient checkpointing for saving memory."
    )
    learning_rate: float = Field(
        default=3e-05,
        description="Learning rate for the optimizer."
    )
    label_smoothing_factor: float = Field(
        default=0.0,
        description="Label smoothing factor for loss calculation."
    )
    bf16: bool = Field(
        default=False,
        description="Whether to use bfloat16 for training."
    )
    fp16: bool = Field(
        default=True,
        description="Whether to use float16 for training."
    )
    optim: str = Field(
        default="adamw_hf",
        description="Optimizer to use for training."
    )
    freeze_w2vm: bool = Field(
        default=False,
        description="Whether to freeze the wav2vec2 model during training."
    )
    save_total_limit: int = Field(
        default=2,
        description="Maximum number of checkpoints to keep during training."
    )
    validation_freq: int = Field(
        default=10,
        description="Frequency of validation steps."
    )
    save_step: int = Field(
        default=10,
        description="Number of steps between saving model checkpoints."
    )
    logging_step: int = Field(
        default=5,
        description="Number of steps between logging training progress."
    )


class TrainConfig(BaseModel):
    """
    Configuration for training
    """

    run_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("RUN_ID", None),
        description="ID associated with a run. Also used to resume a job."
    )

    train_corpora: List[CorpusConfig] = Field(
        ...,
        description="List of training corpora configurations."
    )
    eval_corpora: List[CorpusConfig] = Field(
        ...,
        description="List of evaluation corpora configurations."
    )
    experiment_name: str = Field(
        default="ChaLL",
        description="Name of the experiment."
    )
    experiment_tag: str = Field(
        default="getting_started",
        description="Tag for the experiment to differentiate configurations."
    )
    device: str = Field(
        default="cuda",
        description="Device to run the training on, e.g., 'cuda' or 'cpu'."
    )
    checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a pre-trained model checkpoint to resume training."
    )
    n_valid_samples: int = Field(
        default=-1,
        description="Number of validation samples to use. -1 uses all samples."
    )
    n_train_samples: int = Field(
        default=-1,
        description="Number of training samples to use. -1 uses all samples."
    )
    alt_base_path: str = Field(
        default="./",
        description="Alternate base path for data or checkpoints."
    )
    wav2vec_base_model: str = Field(
        default="facebook/wav2vec2-xls-r-300m",
        description="Base model for wav2vec2 training."
    )
    seed: int = Field(
        default=123,
        description="Random seed for reproducibility."
    )
    vocab_file: Optional[str] = Field(
        default=None,
        description="Path to a vocabulary file. If not provided new one is created."
    )
    train_args: TrainArgs = Field(
        default=TrainArgs(),
        description="Arguments specific to the training process."
    )

    @classmethod
    def from_json(cls, json_path: str) -> "TrainConfig":
        import json
        with open(json_path, 'rt', encoding='utf-8') as ifile:
            config = json.load(ifile)
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    def to_json(self, file_path: str, file_name: str = 'config.json') -> None:
        import os
        import json
        with open(os.path.join(file_path, file_name), 'wt', encoding='utf-8') as out_file:
            json.dump(self.to_dict(), out_file, indent=4)

    @classmethod
    def from_dict(cls, input_dict: dict) -> "TrainConfig":
        """
        Creates an instance of BaseServiceConfig from a dictionary.

        :param input_dict: Dictionary containing the configuration keys and values.
        :return: An instance of the config with values populated from the dictionary.
        """
        return cls.parse_obj(input_dict)

    @classmethod
    def parse_yaml_to_dict(cls, path: str) -> dict:
        """
        Reads and parses the YAML file into a dictionary.

        :param path: Path to the YAML file containing the configuration.
        :return: Parsed dictionary with relevant values.
        """
        with open(path, "r") as file:
            loaded = yaml.safe_load(file)

        data = dict()
        for k, v in loaded.items():
            if isinstance(v, dict) and "value" in v:
                data[k] = v["value"]
            else:
                data[k] = v
        return data

    @classmethod
    def from_yaml(cls, *paths: str, **defaults):
        """
        Creates an instance of BaseServiceConfig from a YAML file.

        :param paths: Paths to YAML files containing the configuration. Later paths override earlier ones.
        :param defaults: Override values that replace those in the YAML file if present.
        :return: An instance of the config with values populated from the YAML file.
        """
        config_data = {}
        for path in paths:
            config_data.update(cls.parse_yaml_to_dict(path))

        config_data.update(defaults)
        return cls.from_dict(config_data)

    @classmethod
    def create_argparser(cls) -> argparse.ArgumentParser:
        """
        Creates an ArgumentParser with arguments based on the Pydantic model fields,
        using each field's type and description for command-line mapping.

        :return: Configured ArgumentParser with model-based arguments.
        """
        parser = argparse.ArgumentParser()
        for name, field in cls.model_fields.items():
            base_type = field.annotation
            if get_origin(base_type) is Union:
                base_type = get_args(base_type)[0]

            parser.add_argument(
                f"--{name}",
                dest=name,
                type=base_type if base_type in {int, float, str, bool} else str,  # Default to str if not callable
                help=field.description,
                default=argparse.SUPPRESS
            )

        # Optional configuration file argument
        parser.add_argument("--config", dest="config", type=str, help="Path to YAML config file")
        return parser

    @classmethod
    def from_cli(cls, **default_dict):
        """
        Initializes the class by merging default values, config file values, and command-line
        arguments, with command-line arguments having the highest priority.

        :param default_dict: Default values for fields in the model.
        :return: An instance of the class with combined configuration settings.
        """
        argparser = cls.create_argparser()
        args_dict, unknown_args = argparser.parse_known_args()
        args_dict = vars(args_dict)

        print(f"Unknown Args: {unknown_args}")

        # Convert string representations of dictionaries to actual dictionaries
        for k, v in args_dict.items():
            parsed_value = try_parse_dict_string(v)
            if parsed_value is not None:
                args_dict[k] = parsed_value

        # Parse YAML config if provided
        config_dict = {}
        if args_dict["config"] is not None:
            try:
                config_dict = cls.parse_yaml_to_dict(args_dict["config"])
            except FileNotFoundError:
                print(f"Warning: Config file '{args_dict['config']}' not found. Using defaults.")

        # Merge dictionaries
        final_dict = {**default_dict, **config_dict, **args_dict}
        return cls(**final_dict)
