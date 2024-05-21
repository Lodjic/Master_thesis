from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from lt_lib.utils.load_and_save import load_yaml_as_dict


class YamlModel(BaseModel):
    @classmethod
    def from_yaml(cls, file_path: str | Path) -> Self:
        """Read a configuration from a yaml file and instantiate a YamlModel.

        Args:
            file_path: path of the yaml file containing the configuration.

        Returns:
            A configuration object.
        """
        file_path = Path(file_path)
        content = load_yaml_as_dict(file_path)
        return cls(**content)


class BaseYamlConfig(YamlModel):
    # Configure processing of extra attributes
    model_config = ConfigDict(extra="ignore")
    # Config type
    config_type: Literal["ModelConfig", "TrainConfig", "EvalConfig"]


class StrOrStrListObject(BaseModel):
    str_object: str | list[str]


class BasePyConfig:
    def __init__(self, object_name: str | list[str]) -> None:
        self.object_name = StrOrStrListObject(object_name).str_object

    def from_py_file(self, file_path: Path) -> Any:
        """Read a python configuration and returns the config object defined in the python file.

        Args:
            file_path (Path): path of the python file containing the configuration.

        Returns:
            The config object (variable, function or class called config): usually a dict.
        """
        file_path = str(Path(file_path))
        pyconfig = SourceFileLoader("pyconfig", file_path).load_module()

        if isinstance(self.object_name, str):
            return {
                self.object_name: getattr(pyconfig, self.object_name) if hasattr(pyconfig, self.object_name) else None
            }
        else:
            return {
                obj_name: getattr(pyconfig, obj_name) if hasattr(pyconfig, obj_name) else None
                for obj_name in self.object_name
            }
