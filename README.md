# Классификация музыкальных жанров

## Цель проекта: 
Обучить модель, которая классифицирует музыкальные композиции по жанрам.

## Датасет:
Набор данных представляет из себя 1000 музыкальные композиции разделенных на 10 жанров.

## Работа с данными:
Извлечение признаков библиотекой librosa, нормализация данных, разделение на обучающую и тестовую выборки.

## Результаты обучения:
![image](https://user-images.githubusercontent.com/64748758/131314684-7b7016d9-943d-4bf4-9307-418aad17469a.png)

## Работа приложения:
#### Запуск в Google Colab:
```ini
!git clone https://github.com/petrov-pa/Music_Genre_Classification.git
%cd ./Music_Genre_Classification
pip install -r requirements.txt
!python main.py
```
Перед запуском необходимо добавить аудио-файлы в папку music.
Результат распознавания записывается в файл result.txt
#### Запуск через Flask:
```ini
git clone https://github.com/petrov-pa/Music_Genre_Classification.git
cd ./Music_Genre_Classification
pip install -r requirements.txt
python run.py
```
#### Запуск через Docker:
```ini
git clone https://github.com/petrov-pa/Music_Genre_Classification.git
cd ./Music_Genre_Classification
pip install -r requirements.txt
docker-compose up --build
```
