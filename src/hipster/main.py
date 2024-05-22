#!/usr/bin/env python3

import argparse
import importlib
import sys

import torch
import yaml

from .hipster import Hipster


def main() -> int:
    list_of_tasks = [
        "hips",
        "catalog",
        "votable",
        "projection",
        "images",
        "thumbnails",
        "allsky",
        "all",
    ]

    parser = argparse.ArgumentParser(
        description="Transform a model in a HiPS representation"
    )
    parser.add_argument(
        "--task",
        "-t",
        nargs="+",
        default=["all"],
        help="Execution task [" + ", ".join(list_of_tasks) + "].",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="config file (default = 'config.yaml').",
    )
    parser.add_argument(
        "--checkpoint",
        "-m",
        default="model.ckpt",
        help="checkpoint file (default = 'model.ckpt').",
    )
    parser.add_argument(
        "--max_order",
        default=4,
        type=int,
        help="Maximal order of HiPS tiles (default = 4).",
    )
    parser.add_argument(
        "--hierarchy",
        default=1,
        type=int,
        help="Number of tiles hierarchically combined (default = 1).",
    )
    parser.add_argument(
        "--crop_size", default=256, type=int, help="Image crop size (default = 256)."
    )
    parser.add_argument(
        "--output_size", default=64, type=int, help="Image output size (default = 64)."
    )
    parser.add_argument(
        "--output_folder",
        default="./HiPSter",
        help="Output of HiPS (default = './HiPSter').",
    )
    parser.add_argument(
        "--title", default="Illustris", help="HiPS title (default = 'Illustris')."
    )
    parser.add_argument(
        "--distortion", action="store_true", help="Enable distortion correction."
    )
    parser.add_argument(
        "--verbose", "-v", default=0, action="count", help="Print level."
    )

    args = parser.parse_args()

    # Check if the tasks are valid
    if not set(args.task) <= set(list_of_tasks):
        raise ValueError(f"Task '{args.task}' not in list of tasks: {list_of_tasks}")

    # If "all" is in the list of tasks, replace it with all tasks except "all"
    if "all" in args.task:
        args.task = list_of_tasks[:-1]

    print(f"Executing task(s): {', '.join(args.task)}")

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # Import the model class and create an instance of it
    model = None
    if any(x in ["hips", "catalog"] for x in args.task):
        model_class_path = config["model"]["class_path"]
        module_name, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        model_init_args = config["model"]["init_args"]
        model = model_class(**model_init_args)
        # FIXME: This is a hack to load the model on the GPU
        # See https://pytorch.org/docs/stable/generated/torch.load.html for map_location
        checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

    # Import the data module and create an instance of it
    datamodule = None
    if any(x in ["catalog", "projection", "images", "thumbnails"] for x in args.task):
        data_class_path = config["data"]["class_path"]
        module_name, class_name = data_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        data_class = getattr(module, class_name)
        data_init_args = config["data"]["init_args"]
        datamodule = data_class(**data_init_args)

    hipster = Hipster(
        args.output_folder,
        args.title,
        max_order=args.max_order,
        hierarchy=args.hierarchy,
        crop_size=args.crop_size,
        output_size=args.output_size,
        distortion_correction=args.distortion,
        catalog_file="catalog.csv",
        votable_file="catalog.vot",
        verbose=args.verbose,
    )

    if "hips" in args.task:
        hipster.generate_hips(model)

    if "catalog" in args.task:
        hipster.generate_catalog(model, datamodule)

    if "votable" in args.task:
        hipster.transform_csv_to_votable()

    if "projection" in args.task:
        hipster.generate_dataset_projection(datamodule)

    if "images" in args.task:
        hipster.create_images(datamodule)

    if "thumbnails" in args.task:
        hipster.create_thumbnails(datamodule)

    if "allsky" in args.task:
        hipster.create_allsky()

    return 0


if __name__ == "__main__":
    sys.exit(main())
