# **Analiza Literatury**

Poniżej znajduje się tabela analizująca wybrane artykuły i źródła danych. Analiza zawiera linki do artykułów, kluczowe wnioski, informacje o dostępności kodu i pre-trenowanych modeli, metryki użyte do ewaluacji oraz wykorzystane zasoby obliczeniowe.

---

## **Tabela analizy literatury**

| **Artykuł/Źródło**                                                                                                                                          | **Wniosek**                                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Temporal Convolutional Networks and Forecasting](https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/)                             | TCN przewiduje jeden krok w czasie, co oznacza, że konieczne jest zapętlenie predykcji w przypadku prognoz wielokrokowych. Ważne jest odpowiednie dobranie **rozmiaru kernela** i **dilacji**, co wpływa na efektywność modelu. Biblioteka **Darts** używana w artykule zapewnia pre-trenowane modele i implementację TCN.                                                    |
| [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)                                                                           | LSTM również przewiduje jeden krok, dlatego konieczne jest zapętlenie predykcji w prognozach wielokrokowych. Model jest skuteczny w sekwencyjnym modelowaniu danych, ale wymaga większych zasobów obliczeniowych niż TCN.                                                        |
| [Deep Learning for Weather Prediction](https://link.springer.com/article/10.1007/s00500-020-04954-0#Sec16)                                                  | W artykule bazowym pokazano, że TCN przewyższa LSTM w predykcjach meteorologicznych. Metryki użyte w ewaluacji: **RMSE (Root Mean Square Error)** i **MAE (Mean Absolute Error)**. Wykorzystane zasoby: GPU do trenowania modeli.                                                         |
| [Meteostat Documentation](https://dev.meteostat.net/)                                                                                                       | Meteostat oferuje zarówno **API**, jak i **bibliotekę** do pracy z danymi meteorologicznymi. Do trenowania modelu można korzystać z danych historycznych dostępnych w bibliotece, a do użytkowania aplikacji w czasie rzeczywistym można wykorzystać API z najnowszymi danymi.                                                             |

---

## **Podsumowanie**

Na podstawie analizy literatury wyciągnięto następujące wnioski:
1. **Temporal Convolutional Network (TCN)**:  
   - Ważne jest odpowiednie dobranie parametrów, takich jak rozmiar kernela i współczynnik dilacji.  
   - Model przewiduje jeden krok w czasie, dlatego konieczne jest zapętlenie prognoz wielokrokowych.  
   - W artykule bazowym TCN przewyższył LSTM w zakresie prognoz meteorologicznych.

2. **Long Short Term Memory (LSTM)**:  
   - Model wymaga większych zasobów obliczeniowych w porównaniu z TCN.  
   - LSTM również przewiduje jeden krok w czasie, więc wymaga zapętlenia przy prognozach wielokrokowych.  

3. **Dane Meteostat**:  
   - Biblioteka Meteostat pozwala na łatwy dostęp do danych historycznych, co jest idealne do trenowania modeli.  
   - API umożliwia wykorzystanie najnowszych danych w aplikacji w czasie rzeczywistym.

---

## **Rekomendacje**

Na podstawie powyższych analiz w projekcie:
- Skupimy się na testowaniu obu architektur (TCN i LSTM), kładąc nacisk na konfigurację parametrów modeli.  
- Do trenowania modeli wykorzystamy bibliotekę Meteostat, a dane z API wykorzystamy w aplikacji do wizualizacji prognoz.  
- Do ewaluacji modeli zastosujemy metryki RMSE i MAE, zgodnie z artykułem bazowym.

