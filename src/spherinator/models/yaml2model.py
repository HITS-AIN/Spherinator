import importlib

import torch
import yaml


def get_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def initialize_class(class_config):
    class_path = class_config["class_path"]
    init_args = class_config["init_args"]
    cls = get_class(class_path)
    for key, value in init_args.items():
        if isinstance(value, list):
            init_args[key] = []
            for item in value:
                if isinstance(item, dict) and "class_path" in item:
                    init_args[key].append(initialize_class(item))
                else:
                    init_args[key].append(item)
        if isinstance(value, dict) and "class_path" in value:
            init_args[key] = initialize_class(value)
    return cls(**init_args)


def yaml2model(yaml_path: str) -> torch.nn.Module:
    """Initialize a PyTorch model from a YAML file"""
    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    return initialize_class(config["model"])
