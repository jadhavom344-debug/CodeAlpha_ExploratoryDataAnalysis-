# CodeAlpha_ExploratoryDataAnalysis

## Task
**CodeAlpha Data Analytics Internship — Task 2: Exploratory Data Analysis (EDA)**

Perform exploratory data analysis on the Titanic passenger dataset to uncover
patterns, trends, and relationships in the data using Python, pandas, and
data visualization libraries.

## Objective
- Load and clean the dataset (handle missing values)
- Generate summary statistics
- Visualize survival patterns across gender, class, age, and fare
- Identify correlations between numeric features

## Dataset
`titanic_sample.csv` — a sample of Titanic passenger records with the
following columns:

| Column | Description |
|---|---|
| PassengerId | Unique passenger ID |
| Survived | 0 = No, 1 = Yes |
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (C/Q/S) |

## Tools & Libraries
- Python 3
- pandas, numpy
- matplotlib, seaborn

## How to Run
```bash
pip install -r requirements.txt
python eda_titanic.py
```

The script prints dataset info and summary statistics to the console and
saves six chart images to the project folder.

## Charts Generated
| File | Description |
|---|---|
| `chart1_survival_count.png` | Overall count of survivors vs non-survivors |
| `chart2_survival_by_gender.png` | Survival split by gender |
| `chart3_survival_by_class.png` | Survival split by passenger class |
| `chart4_age_distribution.png` | Age distribution, stacked by survival |
| `chart5_fare_by_class.png` | Fare distribution across passenger classes |
| `chart6_correlation_heatmap.png` | Correlation heatmap of numeric features |

## Key Insights
- Overall survival rate in the sample was **32%**.
- **Women** had a much higher survival rate (~62%) than **men** (~16%),
  reflecting the "women and children first" evacuation pattern.
- **1st class** passengers survived at a notably higher rate (~49%) than
  2nd (~26%) and 3rd class (~26%) passengers.
- Fare correlates with class: 1st class passengers paid substantially
  higher fares on average.
- Age and Fare show weak-to-moderate correlation with survival, while
  Pclass shows a stronger negative correlation.

## Project Structure
```
CodeAlpha_ExploratoryDataAnalysis/
├── eda_titanic.py
├── generate_data.py
├── titanic_sample.csv
├── requirements.txt
├── chart1_survival_count.png
├── chart2_survival_by_gender.png
├── chart3_survival_by_class.png
├── chart4_age_distribution.png
├── chart5_fare_by_class.png
├── chart6_correlation_heatmap.png
└── README.md
```

## Author
Om Jadhav — CodeAlpha Data Analytics Internship
