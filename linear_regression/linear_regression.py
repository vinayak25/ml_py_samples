
# coding: utf-8

#  # Salary Prediction On basis of Years of Experience using Simple Linear Regression

# In[1]:


#importing primary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Reading Dataset from Sample CSV file
dataset = pd.read_csv('Salary_Data.csv')
#Printing the dataset just for reference.
print(dataset)


# In[3]:


#Distributing the X-axis points, in this case: Years of Experience will be on X-axis.
X = dataset.iloc[:,:-1].values
#Printing the dataset just for reference.
print(X)


# In[4]:


#Distributing the Y-axis points, in this case: Salary will be on Y-axis.
Y = dataset.iloc[:,1].values
#Printing the dataset just for reference.
print(Y)


# ## Splitting into training and test data
# ### Now, that we have distributed the dataset for X-axis and Y-axis. It's time to split them into training and test data.

# In[5]:


#Importing library
#Note that, the "cross_validation" class is depreceated now. Hence using model_selection.
from sklearn.model_selection import train_test_split


# In[6]:


#The function train_test_split will return the splitted data for the same.
#Note that, "test_size" parameter will define the size of the dataset to be used for testing.
#It is recommended to keep the test_size low.
#Also, "random_state" will decide whether or not to randomize the dataset.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 1/3, random_state = 0)


# ### Printing the training dataset.

# In[8]:


print(X_train)


# In[9]:


print(Y_train)


# ### Printing the test dataset.

# In[10]:


print(X_test)


# In[11]:


print(Y_test)


# ## Training the SLR model.

# In[12]:


#Importing library
from sklearn.linear_model import LinearRegression


# In[13]:


#Creating an instance of LinearRegression()
regressor = LinearRegression()


# In[14]:


#Fitting X_train and Y_train dataset into our "regressor" model.
regressor.fit(X_train, Y_train)


# ## After training, time to predict salaries for the test dataset from regressor

# In[15]:


#Predicting the salaries from test dataset.
Y_pred = regressor.predict(X_test)
#Printing Y_pred just for reference
print(Y_pred)


# ## Visualising the training set results

# In[16]:


plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color="green")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# ## Visualising the test set results

# In[17]:


plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

