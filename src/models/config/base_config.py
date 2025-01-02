import argparse
import json
from typing import Dict, Any, get_origin, Union, get_args

import yaml
from pydantic import BaseModel


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


class BaseConfig(BaseModel):

    @classmethod
    def from_json(cls, json_path: str) -> "BaseConfig":
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
    def from_dict(cls, input_dict: dict) -> "BaseConfig":
        """
        Creates an instance of BaseServiceConfig from a dictionary.

        :param input_dict: Dictionary containing the configuration keys and values.
        :return: An instance of the config with values populated from the dictionary.
        """
        return cls.parse_obj(input_dict)

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

    @staticmethod
    def deep_update(original, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                # If both are dicts, update recursively
                BaseConfig.deep_update(original[key], value)
            else:
                # Otherwise, overwrite
                original[key] = value

    @classmethod
    def from_cli(cls, default_config_file: str = None, **default_dict):
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
                if default_config_file is not None:
                    config_dict = cls.parse_yaml_to_dict(default_config_file)
                    cls.deep_update(original=config_dict, updates=cls.parse_yaml_to_dict(args_dict["config"]))
                else:
                    config_dict = cls.parse_yaml_to_dict(args_dict["config"])
            except FileNotFoundError:
                print(f"Warning: Config file '{args_dict['config']}' not found. Using defaults.")

        # Merge dictionaries
        final_dict = {**default_dict, **config_dict, **args_dict}
        return cls(**final_dict)

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
