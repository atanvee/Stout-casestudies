#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Data Description

# In[2]:


# reading the data
data=pd.read_csv('loan_1.csv')


# In[3]:


# getting information of data
data.info()


# We can see that there are 18 float columns , 25 int columns and 13 categorical columns. Also we can observe that a lot of columns have a huge amount of null values present.

# In[4]:


#summary statistics on data
data.describe()


# We can see that the range of values has a huge difference for the columns and we need to standardize all the float and int columns as a result.

# In[5]:


#Checking value count for each column
for item in data:
    print(item,data[item].value_counts())


# Here we can check how many times each value comes for every column.

# In[6]:


#checking for null values
data.isnull().sum()


# We can clearly see that some columns like annual_income_joint,verification_income_joint ,debt_income_joint has huge percentage of null values.Hence these columns can be dropped without loss of a lot of information.

# In[7]:


#dropping columns with more than 20% of null values
data.drop(['Unnamed: 55','verification_income_joint','annual_income_joint','debt_to_income_joint','months_since_90d_late','months_since_last_delinq'],axis=1,inplace=True)


# In[8]:


data.isnull().sum().sort_values(ascending=False)


# In[9]:


# replacing null values with the mode in the respective column
null_columns = data.columns[data.isna().any()].tolist()
for col in null_columns:
    data[col] = data[col].fillna(
    data[col].dropna().mode().values[0] )   

    
data.isnull().sum().sort_values(ascending=False)


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
#plotting correlation heatmap to find the correlated columns
plt.figure(figsize=(12,12))
sns.heatmap(data.corr())
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()


# We can see from the heatmap that there is high positive correlation(lighter colours) between some column and highly positive correlation(darker colours) between the others.

# In[11]:


# plotting distanceplot for occurences of different interest rates
def distanceplot(data):
    from scipy.stats import norm
    sns.distplot(data["interest_rate"], fit=norm)
    plt.title(" Skew of Interest Rate")
    plt.xlabel("Interest Rate")
    plt.ylabel("Occurance ")
    plt.show()
    return

distanceplot(data)


# As we can see the  distance plot is right skewed which means the occurence of low interest rate is more than that of very high interest rates.Also some models work better on normal data so we might have to handle this during data preprocessing.

# In[12]:


# Violin Plot
def violin_plot(data):
    sns.violinplot(x="homeownership", y="interest_rate", data=data, hue="term")
    plt.title("Violin Plot")
    plt.xlabel("Home Ownership")
    plt.ylabel("Interest Rate")
    plt.show()
    return

violin_plot(data)


# The violin plot is the mix of kernel density plot and box plot.We can understand the distribution of the interest rate according to the home ownership of the customers better using the violin plot. 

# In[13]:


def lineplot(data):
    """
    Employment length vs interest rate
    """
    sns.lineplot(x=data['emp_length'], y=data['interest_rate'])
    plt.title("Employment Length vs Interest Rate")
    plt.xlabel("Employment Length in yrs")
    plt.ylabel("Interest Rate")
    plt.show()
    return

lineplot(data)


# In[ ]:





# In[14]:



#defining function to create boxplot
def get_boxplot(data):
    sns.boxplot(x="variable", y="value", data=pd.melt(data))
    sns.set(rc={'figure.figsize':(20,10)})
    plt.figure(figsize=(20,10))
    plt.show()  
    


# In[15]:


# seperating numerical columns and categorical columns and making a boxplot for the numerical columns
y = data["interest_rate"]
data.drop(['interest_rate'],axis=1,inplace=True)
num = data.select_dtypes('number').columns.to_list()
#list of all the categoric columns
obj = data.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  data[num]
#categoric df
loan_obj = data[obj]

get_boxplot(loan_num)


# As it is evident from the boxplot the range of values is different and we need to standardize the columns.

# The key points we can see from the analysis of the data are:-
#    
#    1- The data has 10000 rows and 55 columns which means there are 10000 different loans alloted.
#    2- There is a huge no. of null values in some columns('Unnamed:55','verification_income_joint','annual_income_joint','debt_to_income_joint','months_since_90d_late','months_since_last_delinq) which had to be removed.The data with lesser no. of null values than 20% can be replaced by the mode of the column
#    3- The data has to be scaled as the ranges of values are very different and some features might dominate the other.
#    4- The data is right skewed which might be a problem as some models work better with normalised data.
#    5-There are outliers present in the data which has to be treated
#    

# # Data preprocessing

# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[17]:


#scalling of data columns
data_scaled = scaler.fit_transform(loan_num)


# In[18]:


columns = list(loan_num)
scaled_features_df = pd.DataFrame(data_scaled, index=data.index, columns=columns)
scaled_features_df.head()


# In[19]:


get_boxplot(scaled_features_df)


# As we can see the data is more comparable in the ranges for different columns now so no 1 feature will overdominate.

# In[20]:


def remove_outliers(df, x):
    # Set Limits
    '''q10, q90 = np.percentile(df[x], 10), np.percentile(df[x], 90)
    iqr = q90 - q10
    cut_off = iqr * 1.5
    lower, upper = q10-cut_off ,  q90 + cut_off
    df[df[x]>upper] = upper
    df[df[x]<lower] = lower
    print('Outliers of "{}" are removed\n'.format(x))
    return df'''
    
    q25, q75 = np.percentile(df[x], 25), np.percentile(df[x], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25-cut_off ,  q75 + cut_off
    df[df[x]>upper] = upper
    df[df[x]<lower] = lower
    print('Outliers of "{}" are removed\n'.format(x))
    return df


# In[21]:


import numpy as np
for item in columns:
    scaled_features_df= remove_outliers(scaled_features_df, item)
scaled_features_df.head()


# In[22]:


get_boxplot(scaled_features_df)


# The boxplot shows that most of the outliers have been handled and the range of values for the column is also comparable.

# In[23]:


# using one hot encoding on the categorical variables 
obj_columns = list(loan_obj)
for col in obj_columns:
    print(col + ":" + str(len(loan_obj[col].unique())))
# removing column with more than 10 categories
loan_obj.drop(["emp_title","state","loan_purpose","sub_grade"],axis=1,inplace=True)
loan_obj_hot = pd.get_dummies(loan_obj,drop_first=True)
loan_obj_hot.head()


# In[24]:


clean_data = pd.concat([scaled_features_df,loan_obj_hot],axis=1)
scaled_features_df.head()


# In[25]:


#plotting correlation heatmap
clean_data_tr = pd.concat([clean_data,y],axis=1)
plt.figure(figsize=(12,12))
sns.heatmap(clean_data_tr.corr())
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()


# The heatmap shows the correlation of all the features with each other.For example We can see that our target interest_rate is highly positively correlated with grade_D and highly negative correlated with grade_B

# In[26]:


# Performing PCA to reduce dimensionality of the data
from sklearn.decomposition import PCA

pca = PCA(n_components = 20).fit(clean_data)
print(pca.explained_variance_ratio_.cumsum()) 
data_new = pca.transform(clean_data)


# As we can see 97.8% variance is covered in just 30 pca features.Hence we just need this 30 features to retain a lot of information and get good results.

# # Model Training

# In[27]:


# implementing linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(clean_data,y,test_size=0.2,random_state =42)
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[28]:


y_pred = lr.predict(X_test)


# In[29]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[30]:


# visualising the regression line using the predictions
plt.figure(figsize=(10,10))
sns.regplot(x=y_test, y=y_pred)


# In[31]:


from sklearn.ensemble import RandomForestRegressor
cls = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)
estimators = np.arange(10, 250, 10) # 10 to 250 increased with 10
scores = []
for n in estimators:
    cls.set_params(n_estimators=n)
    cls.fit(X_train, y_train)
    scores.append(cls.score(X_train, y_train))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[32]:


import numpy
from sklearn import linear_model
cls = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)#cls = RandomForestRegressor(n_estimators=150)

cls.fit(X_train, y_train)#We are training the model with RBF'ed data

scoreOfModel = cls.score(X_train, y_train)


print("Score is calculated as: ",scoreOfModel)


# In[33]:


y_pred = cls.predict(X_test)
y_pred


# In[34]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[35]:


# visualising the regression line using the predictions
plt.figure(figsize=(10,10))
sns.regplot(x=y_test, y=y_pred)


# # Assumptions

# Some of the assumptions we have made in the data are:-
# 
# 1- As we are using the linear regression we are assuming that the features are linearly dependent.
# 
# 2- I am using 20 PCA components and assuming that those components cover most of the variance in the data.

# # Model improvements

# 1- Collection of more data is always good to improve performance of the model.
# 
# 2- Ensemble methods like bagging and boosting can be used to improve the model.
# 
# 3- Grid search can be used to find the perfect parameters for the regression models.
# 
# 4- Different parameter can be tried for the models and then use the best performing model.
# 
# 5- Tried different ml models for regression like ridge regression and compare its result with the models we have used.

# In[ ]:




