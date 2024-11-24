# ZPRP-METEO-MODEL

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projekt badawczy mający na celu opracowanie modelu do przewidywania wartości meteorologicznych, takich jak temperatura, ciśnienie i opady. Prognozy będą wizualizowane na prostych wykresach liniowych w dedykowanej aplikacji.

---

## **Opis**

Projekt obejmuje:
- Przegląd literatury naukowej w celu wybrania najlepszych metod.
- Eksperymenty z różnymi architekturami sieci neuronowych, aby wytrenować najdokładniejszy model prognostyczny.
- Wizualizację wyników modelu w oparciu o dane aktualne i historyczne.

Projekt jest inspirowany artykułem [Springer](https://link.springer.com/article/10.1007/s00500-020-04954-0#Sec16), jednak wprowadza zmiany, takie jak:
- Wykorzystanie innego źródła danych: [Meteostat](https://dev.meteostat.net/guide.html).
- Proste zastosowanie modelu na bieżących danych meteorologicznych.

---

## **Kluczowe funkcjonalności**

- **Reprodukcja eksperymentów**: Kod projektu umożliwia odtworzenie eksperymentów przeprowadzonych podczas prac badawczych.  
- **Wizualizacja prognoz**: Prosta aplikacja Streamlit prezentuje:
  - Prognozy temperatury.
  - Przewidywania ciśnienia.
  - Szacowania opadów.

---

## **Instalacja**

Aby uruchomić projekt lokalnie:

1. Sklonuj repozytorium:
   ```bash
   git clone https://gitlab-stud.elka.pw.edu.pl/mostasze/zprp-meteo-model.git
   cd zprp-meteo-model
   ```

2. Utwórz i aktywuj środowisko wirtualne
    ```bash
    make create_environment
    ```

3. Zainstaluj zależności:
    ```bash
    make requirements
    ```

---

## **Sposób użycia**

1. Przygotowanie danych:
    ```bash 
    make prepare_data
    ```

2. Uruchomienie eksperymentów:
    ```bash
    make run_experiments
    ```

---

## Organizacja projektu

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         zprp_meteo_model and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── zprp_meteo_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zprp_meteo_model a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## **Autorzy**

- Michał Sadowski
- Mateusz Ostaszewski
- Szymon Łukawski

---

## **Licencja**

Projekt jest objęty licencją MIT. Szczegóły można znaleźć w pliku [LICENSE](LICENSE). 

---

## **Dodatkowa dokumentacja**

Szczegółowe informacje o projekcie znajdują się w dodatkowych plikach:  
- [Design Proposal](docs/DesignProposal.md)

--------

