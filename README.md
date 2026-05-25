# RR_project

Project comparing:
- Naive
- Seasonal Naive
- AutoARIMA
- LightGBM
- Chronos
- Chronos 2

## Structure

- data/ → datasets
- notebooks/ → experiments
- src/ → source code
- outputs/ → forecasts and metrics
- report/ → final report

- ## Notes for reference
- Modele wybrane na podstawie artykulu na ktorym baszujemy
- light gbm nie byl porownywany przez autorow ale byl wspomniany ze moze byc dobra konkurencja dla chronosa wiec dodalismy go i zrobilismy w ten sposob extended veriosn of the project
- dodalismy tez nowa wersja chronosa - chronos 2, ktorej nie bylo w momencie pisania artykulu

- Podzial zadan:

- Iza - EDA, podzial train/val/test, Naive, Seasonal Naine, AutoArima
- Kasia - Przygotowanie repo i struktury, lagi, rolling features, recursive LightGBM
- Paweł - Chronos 1, Chronos 2, reproducibility, readme
