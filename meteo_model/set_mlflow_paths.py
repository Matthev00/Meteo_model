import os
from meteo_model.utils.mlflow_utils import update_meta_yaml, update_model_meta_yaml


def main():
    old_prefix = "file:///home/mateusz/PW/ZPRP/Projekt/zprp-meteo-model"
    current_directory = os.getcwd()
    new_prefix = f"file://{current_directory}"
    print(f"Updating artifact paths from {old_prefix} to {new_prefix}")
    update_meta_yaml("mlruns", old_prefix, new_prefix)
    update_model_meta_yaml("mlruns/models", old_prefix, new_prefix)


if __name__ == "__main__":
    main()
