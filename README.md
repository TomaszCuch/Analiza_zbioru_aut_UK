# Analiza_zbioru_aut_UK

Analiza zbioru danych na temat samochodów osobowych na terenie UK, dane z października 2022

Na danych wytrenowano modele ML:
- Regresja liniowa
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Sieć neuronowa MLP

Najlepszy okazał się model XGBoost (MAPE ~ 7.4%)

W folderze `app` znajduje się aplikacja zbudowana w pakiecie `shiny` (app.R) do prognozowania ceny wybranego samochodu na podstawie modelu XGBoost.
