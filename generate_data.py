import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

pclass = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
sex = np.random.choice(['male', 'female'], size=n, p=[0.65, 0.35])

age = []
for p in pclass:
    if p == 1:
        age.append(np.clip(np.random.normal(38, 14), 1, 80))
    elif p == 2:
        age.append(np.clip(np.random.normal(30, 13), 1, 75))
    else:
        age.append(np.clip(np.random.normal(25, 12), 0.5, 70))
age = np.round(age, 1)

sibsp = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.62, 0.25, 0.08, 0.03, 0.02])
parch = np.random.choice([0, 1, 2, 3], size=n, p=[0.68, 0.18, 0.10, 0.04])

fare = []
for p in pclass:
    if p == 1:
        fare.append(np.random.gamma(4, 20))
    elif p == 2:
        fare.append(np.random.gamma(3, 7))
    else:
        fare.append(np.random.gamma(2, 4))
fare = np.round(fare, 2)

embarked = np.random.choice(['S', 'C', 'Q'], size=n, p=[0.72, 0.19, 0.09])

# survival probability influenced by class, sex, age (mirrors real historical pattern)
survive_prob = 0.05 + 0.35 * (sex == 'female') + 0.25 * (pclass == 1) + \
               0.10 * (pclass == 2) + 0.15 * (age < 12)
survive_prob = np.clip(survive_prob, 0.02, 0.95)
survived = np.random.binomial(1, survive_prob)

first_names_m = ['James', 'John', 'William', 'Henry', 'George', 'Thomas', 'Charles', 'Frank']
first_names_f = ['Mary', 'Anna', 'Elizabeth', 'Margaret', 'Alice', 'Florence', 'Ellen', 'Kate']
last_names = ['Smith', 'Brown', 'Johnson', 'Williams', 'Jones', 'Miller', 'Davis', 'Wilson',
              'Taylor', 'Anderson', "O'Brien", 'Murphy', 'Kelly', 'Ryan', 'Walsh']

names = []
for s in sex:
    fn = np.random.choice(first_names_m if s == 'male' else first_names_f)
    ln = np.random.choice(last_names)
    title = 'Mr.' if s == 'male' else np.random.choice(['Mrs.', 'Miss.'])
    names.append(f"{ln}, {title} {fn}")

df = pd.DataFrame({
    'PassengerId': np.arange(1, n + 1),
    'Survived': survived,
    'Pclass': pclass,
    'Name': names,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
})

# introduce a few realistic missing values
missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, 'Age'] = np.nan
missing_emb = np.random.choice(df.index, size=2, replace=False)
df.loc[missing_emb, 'Embarked'] = np.nan

df.to_csv('titanic_sample.csv', index=False)
print(df.head())
print(df.shape)
