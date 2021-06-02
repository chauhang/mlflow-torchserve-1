import json
from argparse import ArgumentParser
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import TextFeature, ImageFeature
from captum.insights.attr_vis import server


def get_class_from_module_path(module_path):
    import os, sys
    module_dir, module_file = os.path.split(module_path)
    module_name, module_ext = os.path.splitext(module_file)
    import importlib.util
    import importlib
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module_obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_obj)
    sys.path.append(module_dir)
    importlib.import_module(module_name)
    import inspect
    model_class = None
    for cls in inspect.getmembers(
            module_obj,
            lambda member: inspect.isclass(member)
                           and member.__module__ == module_obj.__name__,
    ):
        model_class = cls[1]
    return model_class


def get_visualizer(data_file_path, visualizer_json_path, model_file_path, wrapper_file_path=None):
    import pickle
    import torch
    f = open(visualizer_json_path, "r")
    data = json.load(f)
    data_handler_class = get_class_from_module_path(data_file_path)
    data_obj = data_handler_class()
    model_class = get_class_from_module_path(model_file_path)
    net = model_class()
    net.load_state_dict(torch.load(data["model"]))
    if wrapper_file_path:
        model_class1 = get_class_from_module_path(wrapper_file_path)
        model_wrapper = model_class1(net)
        net = model_wrapper
        net.eval()

    baseline = pickle.loads(bytes(data["baseline"], "ISO-8859-1"))
    transform = pickle.loads(bytes(data["transform"], "ISO-8859-1"))
    vis_col = pickle.loads(bytes(data["visualization_transform"], "ISO-8859-1")) \
        if "visualization_transform" in data else None

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
        dataset=data_obj.get_batch_data(),
    )


@server.app.route("/visualizer", methods=["POST"])
def change_visualizer():
    from flask import request
    inp = request.get_json(force=True)
    server.visualizer = get_visualizer(data_file_path=inp["data_handler_path"],
                                       visualizer_json_path=inp["visualizer_json_path"],
                                       model_file_path=inp["model_file_path"],
                                       wrapper_file_path=inp["model_wrapper_file"]
                                       if "model_wrapper_file" in inp else None)
    server.visualizer.get_insights_config()
    return "Success"


if __name__ == "__main__":
    parser = ArgumentParser(description="Captum insights")
    parser.add_argument(
        "--data_handler_file",
        type=str,
        required=True,
        help="Path to the python data handler script. Should have get_batch_data() which prepares the data, "
             "creates dataloader and yields captum.insights.Batch eachtime",
    )
    parser.add_argument(
        "--model_file_path",
        type=str,
        required=True,
        help="Full path to the model architecture class file.",
    )
    parser.add_argument(
        "--visualizer_json_path",
        type=str,
        required=True,
        help="Path to the generated json file as string",
    )
    parser.add_argument(
        "--model_wrapper_file",
        type=str,
        default=None,
        help="Full path to the model wrapper class file.",
    )
    args = parser.parse_args()

    server.visualizer = get_visualizer(data_file_path=args.data_handler_file,
                                       visualizer_json_path=args.visualizer_json_path,
                                       model_file_path=args.model_file_path,
                                       wrapper_file_path=args.model_wrapper_file
                                       if "model_wrapper_file" in args else None)
    server.run_app(debug=True)
