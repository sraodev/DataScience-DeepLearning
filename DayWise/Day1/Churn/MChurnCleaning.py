#### Importing Libraries ####

# =============================================================================
# Cleaning the dataset
# =============================================================================


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('churn_data.csv') # Users who were 60 days enrolled, churn in the next 30


# Removing NaN
dataset.isnull().sum()
dataset[dataset.age.isnull()][['age']]


dataset['age']=dataset['age'].fillna(dataset.age.mean())

dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])


dataset.columns

## Correlation with Response Variable
dataset.drop(columns = ['housing', 'payment_type',
                         'registered_phones', 'zodiac_sign',"user","churn"]
).corrwith(dataset.churn).plot.bar(figsize=(20,30),
              title = 'Correlation with Response variable',
              fontsize = 8, rot = 45,
              grid = True)
plt.show()


# Removing Correlated Fields
dataset = dataset.drop(columns = ['app_web_user'])

## Note: Although there are somewhat correlated fields, they are not colinear
## These feature are not functions of each other, so they won't break the model
## But these feature won't help much either. Feature Selection should remove them.

dataset.to_csv('clean_churn_data.csv', index = False)











