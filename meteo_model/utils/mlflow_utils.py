import os
import yaml


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


if __name__ == "__main__":
    old_prefix = "file:///content/"
    new_prefix = "file:///home/mateusz/PW/ZPRP/Projekt/"
    update_meta_yaml("mlruns", old_prefix, new_prefix)
