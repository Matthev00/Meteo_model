LOCATIONS = {
    "WARSAW": (52.23, 21.01),
    "WROCLAW": (51.07, 17.02),
    "POZNAN": (52.24, 16.55),
    "KRAKOW": (50.04, 19.56),
    "BIALYSTOK": (53.07, 23.09),
}

START_YEAR: int = 2004
END_YEAR: int = 2024
BASE_PATH: str = "data/raw/weather_data"
STATIONS_CACHE_DIR: str = "data/cache"
DATASET_START_YEAR: int = 2012
DATASET_END_YEAR: int = 2024
LOCATIONS_NAMES = list(LOCATIONS.keys())

PATHS_TO_DATA_FILES_STR = "data/processed/weather_data/*/*.csv"
PATH_TO_STATS = "data/stats.json"
