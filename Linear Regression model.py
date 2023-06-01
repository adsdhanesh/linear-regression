#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing data
data = pd.read_csv('Raw_Housing_Prices.csv')
data.head()


# In[3]:


data['Sale Price'].describe()


# In[4]:


#distribution of target variable
data['Sale Price'].plot.hist()


# In[5]:


# checking quantiles
q1 = data['Sale Price'].quantile(0.25)
q3 = data['Sale Price'].quantile(0.75)
q1, q3


# In[6]:


#calculating iqr
iqr = q3 - q1
iqr


# In[7]:


upper_limit = q3 + 1.5*iqr
lower_limit = q1 - 1.5*iqr
upper_limit, lower_limit


# In[8]:


# imputing outliers
def limit_imputer(value):
  if value > upper_limit:
    return upper_limit
  if value < lower_limit:
    return lower_limit
  else:
    return value


# In[9]:


data['Sale Price'] = data['Sale Price'].apply(limit_imputer)


# In[10]:


data['Sale Price'].describe()


# In[11]:


data['Sale Price'].plot.hist()


# In[12]:


#checking missing values
data.isnull().sum()


# In[13]:


data['Sale Price'].dropna(inplace=True)
data["Sale Price"].isnull().sum()


# In[14]:


data.info()


# In[15]:


#isolating numerical variables
numerical_columns = ['No of Bathrooms', 'Flat Area (in Sqft)','Lot Area (in Sqft)',
                     'Area of the House from Basement (in Sqft)','Latitude',
                     'Longitude','Living Area after Renovation (in Sqft)']


# In[16]:


#imputing missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])


# In[17]:


data.info()


# # zipcode transform

# In[18]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(data['Zipcode'].values.reshape(-1,1))


# In[19]:


data['Zipcode'].shape


# In[20]:


column = data["Zipcode"].values.reshape(-1,1)
column.shape


# In[21]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
data['Zipcode'] = imputer.fit_transform(column)


# In[22]:


data.info()


# # Other transformations

# In[23]:


data['No of Times Visited'].unique()


# In[24]:


# converting from string to categorical
mapping = {'None' : "0",
           'Once' : '1',
           'Twice' : '2',
           'Thrice' : '3',
           'Four' : '4'}

data['No of Times Visited'] = data['No of Times Visited'].map(mapping)


# In[25]:


data['No of Times Visited'].unique()


# In[26]:


# new variable creation
data['Ever Renovated'] = np.where(data['Renovated Year'] == 0, 'No', 'Yes')


# In[27]:


data.head()


# In[28]:


#manipulating datetime variable
data['Purchase Year'] = pd.DatetimeIndex(data['Date House was Sold']).year


# In[29]:


data['Years Since Renovation'] = np.where(data['Ever Renovated'] == 'Yes',
                                                     abs(data['Purchase Year'] - 
                                                        data['Renovated Year']), 0)


# In[30]:


data.head()


# In[31]:


# dropping redundant variables
data.drop( columns = ['Purchase Year', 'Date House was Sold', 'Renovated Year'], inplace = True)


# In[32]:


data.head()


# # zipcodebin

# In[33]:


data.drop( columns = 'ID', inplace = True)


# In[34]:


data['Condition of the House'].head(10)


# In[35]:


data['Condition of the House'].value_counts()


# In[36]:


data.groupby('Condition of the House')['Sale Price'].mean().plot(kind = 'bar')


# In[37]:


data.groupby('Condition of the House')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[38]:


data.groupby('Waterfront View')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[39]:


data.groupby('Ever Renovated')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# In[40]:


data.groupby('Zipcode',)['Sale Price'].mean().sort_values().plot(kind = 'bar')


# # Linear Regression
# 

# In[41]:


data.dropna(inplace=True)
X = data.drop(columns=['Sale Price'])
Y = data['Sale Price']


# ## variable transformation

# In[42]:


#checking distribution of independent numerical variables
def distribution(data ,var):
  plt.figure(figsize = (len(var)*6,6), dpi = 120)
  for j,i in enumerate(var):
    plt.subplot(1,len(var),j+1)
    plt.hist(data[i])
    plt.title(i)


# In[43]:


numerical_columns = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
       'Age of House (in Years)', 'Latitude', 'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[44]:


for i in numerical_columns:
  X[i] = pd.to_numeric(X[i])


# In[45]:


distribution(X, numerical_columns)


# In[46]:


#removing right skew
def right_skew(x):
  return np.log(abs(x+500))

right_skew_variables = ['No of Bedrooms', 'No of Bathrooms', 'Lot Area (in Sqft)',
       'No of Floors',
       'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
        'Longitude',
       'Living Area after Renovation (in Sqft)',
       'Lot Area after Renovation (in Sqft)',
       'Years Since Renovation']


# In[47]:


for i in right_skew_variables:
  X[i] = X[i].map(right_skew)

# removing infinite values
X = X.replace(np.inf, np.nan)
X.dropna(inplace=True)


# In[48]:


distribution(X, numerical_columns)


# ## Scaling the dataset

# In[49]:


X.head()


# In[50]:


X["Waterfront View"] = X["Waterfront View"].map({    'No':0,
   'Yes':1
})


X['Condition of the House'] = X['Condition of the House'].map({'Bad':1,
                                                                     'Okay':2,
                                                                     'Fair':3,
                                                                     'Good':4,
                                                                     'Excellent':5
})

X['Ever Renovated'] = X['Ever Renovated'].map({
    'No':0,
    'Yes':1
})

X.head()


# In[51]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Y = data['Sale Price']
X1 = scaler.fit_transform(X)
X = pd.DataFrame(data = X1, columns = X.columns)
X.head()


# ## Checking and Removing Multicollinearity

# In[52]:


X.corr()


# In[53]:


## pair of independent variables with correlation greater than 0.5
k = X.corr()
z = [[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j] >abs(0.5))&(i!=j)]
z, len(z)


# ### Calculating VIF

# In[54]:


# Importing Variance_inflation_Factor funtion from the Statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = X[:]

## Calculating VIF for every column
VIF = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index = vif_data.columns)
VIF


# In[55]:


def MC_remover(data):
  vif = pd.Series([variance_inflation_factor(data.values, i) for i in range(data.shape[1])], index = data.columns)
  if vif.max() > 5:
    print(vif[vif == vif.max()].index[0],'has been removed')
    data = data.drop(columns = [vif[vif == vif.max()].index[0]])
    return data
  else:
    print('No Multicollinearity present anymore')
    return data


# In[56]:


for i in range(7):
  vif_data = MC_remover(vif_data)

vif_data.head()


# ### Remaining Columns

# In[57]:


# Calculating VIF for remaining columns
VIF = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index = vif_data.columns)
VIF, len(vif_data.columns)


# In[58]:


X = vif_data[:]


# ## Train/Test set

# In[59]:


Y = data['Sale Price']


# In[60]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ## training model

# In[61]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)


# In[62]:


lr.coef_


# In[63]:


predictions = lr.predict(x_test)


# In[64]:


lr.score(x_test, y_test)


# ### 1. Residuals

# In[65]:


residuals = predictions - y_test

residual_table = pd.DataFrame({'residuals':residuals,
                    'predictions':predictions})
residual_table = residual_table.sort_values( by = 'predictions')


# In[66]:


z = [i for i in range(int(residual_table['predictions'].max()))]
k = [0 for i in range(int(residual_table['predictions'].max()))]


# In[67]:


plt.figure(dpi = 130, figsize = (17,7))

plt.scatter( residual_table['predictions'], residual_table['residuals'], color = 'red', s = 2)
plt.plot(z, k, color = 'green', linewidth = 3, label = 'regression line')
plt.ylim(-800000, 800000)
plt.xlabel('fitted points (ordered by predictions)')
plt.ylabel('residuals')
plt.title('residual plot')
plt.legend()
plt.show()


# ## 2. Distribution of errors

# In[68]:


plt.figure(dpi = 100, figsize = (10,7))
plt.hist(residual_table['residuals'], color = 'red', bins = 200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('distribution of residuals')
plt.show()


# ## Model Coefficients

# In[69]:


coefficients_table = pd.DataFrame({'column': x_train.columns,
                                  'coefficients': lr.coef_})
coefficient_table = coefficients_table.sort_values(by = 'coefficients')


# In[70]:


plt.figure(figsize=(8, 6), dpi=120)
x = coefficient_table['column']
y = coefficient_table['coefficients']
plt.barh( x, y)
plt.xlabel( "Coefficients")
plt.ylabel('Variables')
plt.title('Normalized Coefficient plot')
plt.show()

