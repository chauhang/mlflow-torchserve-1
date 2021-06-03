import json
from argparse import ArgumentParser

from captum.insights import AttributionVisualizer
from captum.insights.attr_vis import server
from captum.insights.attr_vis.features import TextFeature, ImageFeature


def get_class_from_module_path(module_path):
    import os
    import sys
    module_dir, module_file = os.path.split(module_path)
    module_name, module_ext = os.path.splitext(module_file)
    import importlib
    sys.path.append(module_dir)
    module_obj = importlib.import_module(module_name)
    import inspect
    model_class = None
    for cls in inspect.getmembers(
            module_obj,
            lambda member: inspect.isclass(member)
                           and member.__module__ == module_obj.__name__,
    ):
        model_class = cls[1]
    return model_class


def get_visualizer(visualizer_json_path):
    import torch
    f = open(visualizer_json_path, "r")
    data = json.load(f)
    model_file_path = data["model_file_path"]
    model_class = get_class_from_module_path(model_file_path)
    net = model_class()
    net.load_state_dict(torch.load(data["model"]))
    if "model_wrapper_file_path" in data:
        model_class1 = get_class_from_module_path(data["model_wrapper_file_path"])
        model_wrapper = model_class1(net)
        net = model_wrapper
        net.eval()
    if "baseline" in data:
        baseline_class = get_class_from_module_path(data["baseline"]["module_path"])
        if hasattr(baseline_class, data["baseline"]["function_name"]):
            baseline = getattr(baseline_class, data["baseline"]["function_name"])
        else:
            raise Exception("Baseline function is not found in specified module")
    else:
        raise Exception("Baseline function is not provided")
    if "transform" in data:
        transform_class = get_class_from_module_path(data["transform"]["module_path"])
        if hasattr(transform_class, data["transform"]["function_name"]):
            transform = getattr(transform_class, data["transform"]["function_name"])
        else:
            raise Exception("Transform function is not found in specified module")
    else:
        raise Exception("Transform function is not provided")
    vis_col = None
    if "visualization" in data:
        vis_class = get_class_from_module_path(data["visualization"]["module_path"])
        if hasattr(vis_class, data["visualization"]["function_name"]):
            vis_col = getattr(vis_class, data["visualization"]["function_name"])
        else:
            raise Exception("Visualization function is not found in specified module")

    if "data_batch_iterator" in data:
        iterator_class = get_class_from_module_path(data["data_batch_iterator"]["module_path"])
        if hasattr(iterator_class, data["data_batch_iterator"]["function_name"]):
            get_batch_data = getattr(iterator_class, data["data_batch_iterator"]["function_name"])
        else:
            raise Exception("Data iterator function is not found in specified module")
    else:
        raise Exception("Transform function is not provided")

    feature = None
    if data["feature_type"] == "TextFeature":
        feature = TextFeature(
            "Text columns",
            baseline_transforms=[baseline],
            input_transforms=[transform],
            visualization_transform=vis_col
        )
    else:
        feature = ImageFeature(
            "Photo",
            baseline_transforms=[baseline],
            input_transforms=[transform],
        )
    return AttributionVisualizer(
        models=[net],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=data["classes"],
        features=[feature],
        dataset=get_batch_data(),
    )


@server.app.route("/visualizer", methods=["POST"])
def change_visualizer():
    from flask import request
    inp = request.get_json(force=True)
    server.visualizer = get_visualizer(visualizer_json_path=inp["visualizer_json_path"])
    server.visualizer.get_insights_config()
    return "Success"


if __name__ == "__main__":
    parser = ArgumentParser(description="Captum insights")
    parser.add_argument(
        "--visualizer_json_path",
        type=str,
        required=True,
        help="Path to the generated json file as string",
    )
    args = parser.parse_args()

    server.visualizer = get_visualizer(visualizer_json_path=args.visualizer_json_path)
    server.run_app(debug=True)
