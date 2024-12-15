import os
import yaml
import shutil


MODELS_DIR = "mlruns/models"
MLRUNS_DIR = "mlruns"


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


def get_valid_run_ids(models_dir):
    valid_run_ids = set()
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file == "meta.yaml":
                meta_path = os.path.join(root, file)
                with open(meta_path, "r") as f:
                    data = yaml.safe_load(f)
                    run_id = data.get("run_id")
                    if run_id:
                        valid_run_ids.add(run_id)
    return valid_run_ids


def delete_invalid_runs(mlruns_dir, valid_run_ids):
    for experiment in os.listdir(mlruns_dir):
        if experiment == "models":
            continue
        experiment_path = os.path.join(mlruns_dir, experiment)
        if os.path.isdir(experiment_path) and experiment != "meta.yaml":
            for run_id in os.listdir(experiment_path):
                run_path = os.path.join(experiment_path, run_id)
                if os.path.isdir(run_path) and run_id not in valid_run_ids:
                    shutil.rmtree(run_path)


if __name__ == "__main__":
    valid_run_ids = get_valid_run_ids(MODELS_DIR)
    delete_invalid_runs(MLRUNS_DIR, valid_run_ids)
