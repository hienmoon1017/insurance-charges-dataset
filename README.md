# Insurance Charges Dataset
I will use Python to perform data analytics on an insurance charges dataset.
### Objectives:
- Load the data as a pandas dataframe
- Clean the data, taking care of the blank entries
- Run exploratory data analysis and identify the attributes that most affect the charges
- Develop single variable and multi variable Linear Regression models for predicting the charges
- Use Ridge regression to refine the performance of Linear regression models.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

file_path = r"D:\Users\medical_insurance_dataset.csv"
df = pd.read_csv(file_path, header=None)
```
## 1. Data Preparation
```python
**# insert headers**
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers
print(df.head(10))
df.info()
```
**- First 10 rows of dataframe:**

![image](https://github.com/user-attachments/assets/0a4120e5-8471-43c0-804d-861c6097989a)

**- Datatypes of all columns:**

![image](https://github.com/user-attachments/assets/6a327a28-3855-4bb2-954e-c31cf17fc0d6)

```python
**# Find and Remove duplicated rows**
print(df[df.duplicated()]) # Result: 1424 duplicated rows

**# Drop duplicates and crosscheck result after removing**
df.drop_duplicates(inplace=True)
df.info() # Result: 1348 remaining rows after removing duplicated rows
```

![image](https://github.com/user-attachments/assets/57b72afd-2d71-4c84-b53e-2e5135dd16b0)

**Comment:**
- 1,348 remaining rows after removing duplicated rows
- data type of "age" and "smoker" are object while they should be int64

**Using Box Plot to find unexpected values in "age" & "smoker" columns**
```python
sns.boxplot(data=df["age"])
sns.boxplot(data=df["smoker"])
plt.tight_layout()
plt.show()
plt.close()
# Result: find out "age" and "smoker" have "?" values
```

![image](https://github.com/user-attachments/assets/8be005f3-9ffd-4d84-9747-ef2cb3dddf99)


![image](https://github.com/user-attachments/assets/a21c2dab-209d-497f-a5b0-a44424352426)

**Replace the "?" entries with "NaN" values**
```python
df.replace("?", np.nan, inplace=True)
```
**Replace missing values with the mean for "age"**
```python
mean_age = df["age"].astype("float").mean(axis=0)
df.loc[:,"age"] = df["age"].replace(np.nan, mean_age) 
```
**Replace missing values with the mode for "smoker"**
```python
mode_smoker = df["smoker"].astype("float").value_counts().idxmax()
df.loc[:,"smoker"] = df["smoker"].replace(np.nan, mode_smoker)
```
**Update the data types**
```python
df[["age","smoker"]] = df[["age","smoker"]].astype("int")
```
**Rounded to nearest 2 decimal places for "charges"**
```python
df.loc[:,"charges"] = df["charges"].round(2)
```
**Verify the updated data types and replace missing values**
```python
df.info()
print(df.head(5))
```
and now I have the cleaned data:

![image](https://github.com/user-attachments/assets/7111dd14-52cc-45e0-a818-facd7a54931f)

_here are the first 5 rows:_

![image](https://github.com/user-attachments/assets/c7c04ec6-c773-420a-a1c6-d49bc7ae70d6)

