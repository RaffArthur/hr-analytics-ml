<img width="1920" height="600" alt="hr-analytics-ml" src="https://github.com/user-attachments/assets/43a773f4-a718-432b-8723-c4d55608c397"/>

***
[![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-FF6B35?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python_3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

[![NumPy](https://img.shields.io/badge/NumPy-✓-3776AB?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-✓-3776AB?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-✓-3776AB?logo=scipy&logoColor=white)](https://scipy.org/)
[![Phik](https://img.shields.io/badge/Phi__K-✓-3776AB?logo=phik&logoColor=white)](https://phik.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-✓-3776AB?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-✓-3776AB?logo=optuna&logoColor=white)](https://optuna.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-✓-3776AB?logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-3776AB?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![SHAP](https://img.shields.io/badge/SHAP-✓-3776AB?logo=shap&logoColor=white)](https://shap.readthedocs.io/)
***

# Разработка предсказательных ML-моделей для отдела HR-аналитики
По требованию заказчика был проведен комплексный исследовательский анализ данных по уровню удовлетворенности сотрудников и вероятности их увольнения. Для исследования и разработки моделей заказчиком были предоставлены общие данные по сотрудникам, данные об удовлетворенность сотрудников условиями труда и данные о фактах увольнения сотрудников

На основе предоставленных данных были разработаны две прогностические ML-модели:
1. Прогнозирование уровня удовлетворённости сотрудников
2. Предсказания вероятности увольнения сотрудника

## Описание данных
Наборы данных:
- `test_features.csv` - общие данные по сотрудникам
- `train_job_satisfaction_rate.csv`, `test_target_job_satisfaction_rate.csv` - удовлетворенность сотрудников условиями труда
- `train_quit.csv`, `test_target_quit.csv` - данные о фактах увольнения сотрудников

Структура данных:
- `id` - уникальный идентификатор сотрудника
- `dept` - отдел, в котором работает сотрудник
- `level`  уровень занимаемой должности
- `workload` - уровень загруженности сотрудника
- `employment_years` - длительность работы в компании (в годах)
- `last_year_promo` - показывает, было ли повышение за последний год
- `last_year_violations` - показывает, нарушал ли сотрудник трудовой договор за последний год
- `supervisor_evaluation` - оценка качества работы сотрудника, которую дал руководитель
- `salary` - ежемесячная зарплата сотрудника
- `job_satisfaction_rate` - уровень удовлетворенности сотрудника работой в компании, целевой признак
- `quit` - увольнение сотрудника из компании

## Результаты
### Обучение ML-моделей

| Задача                        | Критерий успеха (метрика) | Тип поиска       | Модель                                                                   | Значение метрики      |
| ----------------------------- | --------------- | ---------------- | ------------------------------------------------------------------------ | ------------- |
| Удовлетворенность сотрудников | SMAPE ≤ 15      | Optuna           | `SVR(C=3.349629045654751)`                                               | **`13.7046`** |
| Отток сотрудников             | ROC-AUC ≥ 0.91  | RandomizedSearch | `SVC(C=10, class_weight='balanced', gamma='auto', probability=True)`     | **`0.9230`**  |


### SHAP & Permutation Importance анализы
#### Удовлетворённость условиями труда

###### Результаты Permutation Importance
| Признак                    | Permutation Importance |
| -------------------------- | -------- |
| num__salary                | 43.4422  |
| ord__level                 | 33.8021  |
| ord__workload              | 26.0163  |
| num__supervisor_evaluation | 25.0429  |
| num__employment_years      | 11.3187  |

###### Результаты SHAP-анализа
<table>
  <tr>
    <td style="padding: 50px;">
        <img src="https://github.com/user-attachments/assets/91c44d6d-8beb-430c-8379-71ed4506f9eb" height='510' style="display: block;" />
    </td>
    <td style="padding: 5px;">
        <img src="https://github.com/user-attachments/assets/8babe3dd-3b4c-45d6-83c2-051bfa40d1f5" height='600' style="display: block;" />
    </td>
  </tr>
</table>


#### Вероятность увольнения
###### Результаты Permutation Importance
| Признак                    | Permutation Importance |
| ----------------------------- | -------- |
| ord__level                    | 0.1288   |
| remainder__jsr_predict        | 0.1051   |
| num__employment_years         | 0.0523   |
| ord__workload                 | 0.0352   |
| num__supervisor_evaluation    | 0.0048   |

###### Результаты SHAP-анализа
<table>
  <tr>
    <td style="padding: 5px;">
        <img src="https://github.com/user-attachments/assets/525d5327-f8b8-4017-8cfc-055b8234ff59" height='510' width="100%" style="display: block;"/>
    </td>
    <td style="padding: 5px;">
        <img src="https://github.com/user-attachments/assets/1f662237-23c7-4471-8037-1aa03f9308bb" height='600' width="100%" style="display: block;"/>
    </td>
  </tr>
</table>

***

## Структура репозитория
```bash
hr-analytics-ml/
│
├── train_job_satisfaction_rate.csv       # Тренировочная выборка: удовлетворенность
├── train_quit.csv                        # Тренировочная выборка: отток
├── test_features.csv                     # Признаки тестовой выборки
├── test_target_job_satisfaction_rate.csv # Целевой признак: JSR (тест)
├── test_target_quit.csv                  # Целевой признак: quit (тест)
├── hr-analytics-ml.ipynb                 # Основной ноутбук с анализом
├── hr-analytics-ml.py                    # Конвертированный Python-скрипт
├── requirements.txt                      # Зависимости проекта
├── README.md                             # Описание проекта
├── LICENSE                               # Лицензия
└── .gitignore                            # Исключения для Git
```

## Запуск
Клонирование репозитория
```bash
git clone https://github.com/RaffArthur/hr-analytics-ml.git

cd hr-analytics-ml
```

Установка зависимостей
```bash
pip install -r requirements.txt
```

Запуск проекта
```bash
jupyter notebook hr-analytics-ml.ipynb
```
