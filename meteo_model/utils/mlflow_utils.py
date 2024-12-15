import os
import yaml
from meteo_model.utils.model_utils import load_model
import torch


def update_meta_yaml(mlruns_path: str, old_prefix: str, new_prefix: str) -> None:
    for root, dirs, files in os.walk(mlruns_path):
        for file in files:
            if file == "meta.yaml":
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    try:
                        meta_data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        print(f"Error {file_path}: {e}")
                        continue

                    if "artifact_uri" in meta_data:
                        old_path = meta_data["artifact_uri"]
                        new_path = old_path.replace(old_prefix, new_prefix)
                        meta_data["artifact_uri"] = new_path

                        with open(file_path, "w") as f:
                            yaml.safe_dump(meta_data, f)


def move_models_to_project_dir():
    model_for_days = {
        1: ("MeteoModel-1_day", 2),
        2: ("MeteoModel-2_days", 1),
        3: ("MeteoModel-3_days", 1),
        4: ("MeteoModel-4_days", 1),
        5: ("MeteoModel-5_days", 1),
        6: ("MeteoModel-6_days", 2),
        7: ("MeteoModel-7_days", 2),
        8: ("MeteoModel-8_days", 2),
    }

    for n_days, (model_name, v) in model_for_days.items():
        model = load_model(model_name, v)
        model_path = "models/" + model_name
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path + ".pth")
    

if __name__ == "__main__":

    move_models_to_project_dir()
