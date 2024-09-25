# Using Python for Data Analytics on the Insurance Charges Dataset
The dataset file is attached in this repository.

### Data Dictionary:
| Parameter       | Description                             | Content type           |
|-----------------|-----------------------------------------|------------------------|
| age             | Age in years                            | integer                |
| gender          | Male or Female                          | integer (1 or 2)       |
| bmi             | Body mass index                         | float                  |
| no_of_children  | Number of children                      | integer                |
| smoker          | Whether smoker or not                   | integer (0 or 1)       |
| region          | Which US region - NW, NE, SW, SE        | integer (1, 2, 3 or 4) |
| charges         | Annual Insurance charges in USD         | float                  |

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
# insert headers
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
# Find and Remove duplicated rows
print(df[df.duplicated()]) # Result: 1424 duplicated rows

# Drop duplicates and crosscheck result after removing
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

## 2. Exploratory Data Analysis (EDA)

**The correlation matrix for the dataset**

![image](https://github.com/user-attachments/assets/b1d25afa-da01-4e4f-ab20-9771fb815122)

**Implement the box plot for "charges" with respect to "smoker"**

```python
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Smoker vs Charges")
plt.xticks([0, 1], ['Smoker', 'Non Smoker']) # Set xticks to the actual values: 1 for Smoker, 2 for Non Smoker
plt.tight_layout()
plt.ylim(0,)
plt.show()
plt.close()
```
![image](https://github.com/user-attachments/assets/a8c1495c-4088-430f-b5e3-6b6b46950ea2)

## 3. Model Development

**Fit a linear regression model that may be used to predict the "charges" value, just by using the "smoker" attribute of the dataset. Calculate the R^2 score**
```python
from sklearn.linear_model import LinearRegression
X = df[["smoker"]]
Y = df[["charges"]]
lm = LinearRegression()
lm.fit(X,Y)
Yhat1 = lm.predict(X)
r_square1 = lm.score(X,Y)
print("R^2:",r_square1)
```
_Result: R^2: **0.6171512998519898**_

**Fit a linear regression model that may be used to predict the charges value, just by using all other attributes of the dataset. Calculate the R^2 score**
```python
from sklearn.linear_model import LinearRegression
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
Y = df[["charges"]]
lm = LinearRegression()
lm.fit(Z,Y)
Yhat2 = lm.predict(Z)
r_square2 = lm.score(Z,Y)
print("R^2 of charges and all attributes:", r_square2)
```
_Result: R^2: **0.748235378955879** that is better_

**Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create a model that can predict the charges value using all the other attributes of the dataset. Calculate the R^2 score**
```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
Y = df[["charges"]]

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
r_square3 = r2_score(Y,ypipe)
print("R^2 of training pipline:", r_square3)
```
_Result: R^2 of training pipeline: **0.8436089618842255** that is good R^2_

## 4. Model Refinement
**Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Z & Y hold same values as in previous cells
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
Y = df[["charges"]]

x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size=0.20, random_state=1)
```
**Initialize a Ridge regressor that used hyperparameter alpha=0.1. Fit the model using training data data subset. Calculate the R^2 score for the testing data**
```python
# x_train, x_test, y_train, y_test hold same values as in previous cells
from sklearn.linear_model import Ridge

RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
yhat = RigeModel.predict(x_test)

r_square4 = r2_score(y_test, yhat)
print("R^2 of the testing data:", r_square4)
```
_R^2 of the testing data: **0.7417321259315346**_

**Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model, as above, using the training subset. Calculate the R^2 score for the testing subset.**
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
# x_train, x_test, y_train, y_test hold same values as in previous cells

pr = PolynomialFeatures(degree=2) 

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
yhat_pr = RigeModel.predict(x_test_pr)

r_square5 = r2_score(y_test, yhat_pr)
print("R^2 of the testing subset:", r_square5)
```
_R^2 of the testing subset: **0.8343913869992748**_

## Conclusion:
With good R^2 = 83.43% I've created a training pipeline using StandardScaler(), PolynomialFeatures(), and LinearRegression() to predict charges, the next step is to deploy and use this trained model in real-work scenarios.

Thank you for stopping by, and I'm pleased to connect with you, my new friend!

**Please do not forget to FOLLOW and star ⭐ the repository if you find it valuable.**

I wish you a day filled with happiness and energy!

Warm regards,

Hien Moon
