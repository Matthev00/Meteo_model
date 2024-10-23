# Model meteorologiczny

## Funckjonalność programu
Projekt ma charakter głównie researchowy. Planujemy zapoznać się z literaturą oraz przeprowadzić serię eksperymentów w celu wytytrenowania najlepszego modelu. Po uzykaniu zadowalającego modelu zostanie on wykorzystany do przewidywania przyszłych wartości meteorologicznych. Predykcje modelu zostaną zwizualizowane na prostych wykresach liniowych. Dane zbierzemy z [meteostat](https://dev.meteostat.net/guide.html).  
Pomysł projektu bazuję na artykule <https://link.springer.com/article/10.1007/s00500-020-04954-0#Sec16>, ale wprowadza parę zmian. Planujemy zmienić źródło danych oraz nasz projekt zakłada proste użycie modelu dla aktualnych danych.

## Planowy zakres eksperymentów
Planujemy przetestować natępujące architektóry sieci neuronowych:
 - Temporal Convolutional Network (TCN)
   - Romiar wielkiości filtrów
   - Współczynnik rozszerzenia
 - Long Short Term Memory (LSTM)
   - Liczba warstw LST
   - Rozmiar okna czasowego  
  
Dla każdego z eksperymentów przetestowany zostanie również różny zakres sekwnecji wejścia tzn. Dane barane z:
- przed paru dni (np.16)
- przed paru dni i z przed roku
- przed paru dni, z przed roku oraz z przed dwóch lat
  
## Stack technologiczny
**Pytorch** - implementacje modeli, trenowanie modeli  
**MlFlow** - śledzenie eksperymentów numerycznych  
**Pandas, Numpy** - przygotowanie i obróbka danych  
**Meteostat** - pozyskiwania danych  
**Google Colab** - trenowania modeli   
**MatPlotLib** - wizualizacja

## Bibliografia

<https://link.springer.com/article/10.1007/s00500-020-04954-0#Sec16>  
<https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/>   
<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>   
<https://dev.meteostat.net/>  

## Harmonogram
| Tydzień | Planowane postepy | Kamień Milowy|
| --------|-------------------| -------------|
| **4.** (21.10 - 27.10) | Wstepne zapoznanie się z zagadnieniami związanymi z projektem, Design Proposal | **Design Proposal**|
 **5.** (28.10 - 03.11) | Ustawienie środowiska pracy(szkielet projektu), Dokladne zapoznanie się z literaturą | - |
| **6.** (04.11 - 10.11) | Zgromadzenie danych oraz weryfikacja ich poprawności|-|
| **7.** (11.11 - 17.11) | Analiza danych, Przygotowanie DataLoader'ów |-|
| **8.** (18.11 - 24.11) | Konfiguracja środowiska eksperymantalnego, Przygotowanie eksperymenów|**Postęp analizy literaturowej**|
| **9.** (25.11 - 01.12) | Napisanie TCN, Przygotowanie LSTM | - |
| **10.** (02.12 - 08.12) | Trenowanie modeli, Przejrznie raportu MlFlow, Wybranie Modelu |-|
| **11.** (09.12 - 15.12) | Strojenie wybranego modelu, Podstawowa wersja wizualizacji przewidywań                 |-|
| **12.** (16.12 - 22.12) | Dostosowanie aplikacji do danych aktualnych z API|**Funkcjonalna wersja projektu**|
| **13.** (30.12 - 05.01) | Rozwinięcie aplikacji |-|
| **14.** (06.01 - 12.01) |Nagranie filmu prezentacyjnego|**Finalna wersja Projektu**|
