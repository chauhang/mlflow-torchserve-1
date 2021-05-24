import json

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import TextFeature
from captum.insights.attr_vis import server
from captum.insights.attr_vis.config import (
    ATTRIBUTION_METHOD_CONFIG,
    ATTRIBUTION_NAMES_TO_METHODS,
)


class DummyVisualizer:
    def get_insights_config(self):
        print("config .....")
        return {
            "classes": ["classes"],
            "methods": list(ATTRIBUTION_NAMES_TO_METHODS.keys()),
            "method_arguments": server.namedtuple_to_dict(
                {k: v.params for (k, v) in ATTRIBUTION_METHOD_CONFIG.items()}
            ),
            "selected_method": "self._config.attribution_method",
        }


@server.app.route("/visualizer", methods=["POST"])
def change_visualizer():
    import pickle
    import torch
    from flask import request
    inp = request.get_json(force=True)
    f = open(inp["visualizer_json_path"], "r")
    data = json.load(f)
    net = pickle.loads(bytes(data["model_class"], "ISO-8859-1"))
    net.load_state_dict(torch.load(data["model"]))
    baseline = pickle.loads(bytes(data["baseline"], "ISO-8859-1"))
    transform = pickle.loads(bytes(data["transform"], "ISO-8859-1"))
    vis_col = pickle.loads(bytes(data["visualization_transform"], "ISO-8859-1"))

    def formatted_data_iter():
        dataloader = iter(pickle.loads(bytes(data["loader"], "ISO-8859-1")))
        while True:
            images, labels = next(dataloader)
            yield Batch(inputs=images, labels=labels)

    server.visualizer = AttributionVisualizer(
        models=[net],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=data["classes"],
        features=[
            TextFeature(
                "Text columns",
                baseline_transforms=[baseline],
                input_transforms=[transform],
                visualization_transform=vis_col
            )
        ],
        dataset=formatted_data_iter(),
    )
    server.visualizer.get_insights_config()
    return "Success"


if __name__ == "__main__":
    server.visualizer = DummyVisualizer()
    server.run_app(debug=True)
