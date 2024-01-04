#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df_train = pd.read_excel(r'D:\data analysis\project\Ticket fare\Data_Train.xlsx')
df_train.head(4)

df_train.info()


# In[4]:
#to know the total amount of null values in each column
df_train.isnull().sum()
#Removing the null values
df_train.dropna(inplace=True)


data = df_train.copy()
data_1 = df_train.copy()

# In[8]:Feature Transformation
#Separating Date_of_Journey into Date,Month,Year

#change of the object datatype to datetime datatype
def change_into_date_mon_year(col):
    data_1[col] = pd.to_datetime(data_1[col])
    
for feature in ['Dep_Time','Arrival_Time','Date_of_Journey']:
     change_into_date_mon_year(feature)
        
data_1['Journey_day'] = data_1['Date_of_Journey'].dt.day
data_1['Journey_month'] = data_1['Date_of_Journey'].dt.month
data_1['Journey_year'] = data_1['Date_of_Journey'].dt.year


# In[10]:
#Clean dep_time and arrival time and extract the derived attribute
def extract_hour_min(data_1, col):
    data_1[col + "_hour"] = data_1[col].dt.hour
    data_1[col + "_min"] = data_1[col].dt.minute
    return data_1

extract_hour_min(data_1, "Dep_Time")
extract_hour_min(data_1, "Arrival_Time")


# In[11]:
#plot the graph to showcase the frequency of flight departure during a day
def flight_dep_time(x):
    if (x>4) and (x<=8):
        return 'Early-Morning'
    elif (x>8) and (x<=12):
        return 'Morning'
    elif (x>12) and (x<=16):
        return 'Afternoon'
    elif (x>16) and (x<=20):
        return 'Evening'
    else:
        return 'Night'
    
data_1['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind = 'bar')
    
# In[12]:
#creation of interactive graphs
import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot,iplot,init_notebook_mode,download_plotlyjs

init_notebook_mode(connected = True)
cf.go_offline()
data_1['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = 'bar')

#Process the Duration data and create seprate features from Duration feature
data_1['Duration']
#all the data should be in the same format in order to be processed
def processed_duration(x):
    if 'h' not in x:
        x = '0h'+' '+x
    elif 'm' not in x:
        x = x +' '+'0m'
    return x

data_1['Duration'] = data_1['Duration'].apply(processed_duration)

#spliting and silicing
data_1['Duration_hour'] = data_1['Duration'].apply(lambda x:int(x.split(' ')[0][0:-1]))
data_1['Duration_min'] = data_1['Duration'].apply(lambda x:int(x.split(' ')[1][0:-1]))


#Analyse whether duration impacts on price or not
#convert duration into total minutes
data_1['Total_duration(min)'] = data_1['Duration_hour']*60 + data_1['Duration_min']*1

# In[18]:
import plotly.express as px
fig_sc = px.scatter(data_1, x ='Total_duration(min)', y = 'Price', title="Interactive scatterplot", color = 'Total_Stops')
fig_sc.show()


#price analysis of airways
sns.boxplot(x ='Airline', y ='Price', data = data.sort_values('Price', ascending=False))
plt.xticks(rotation = 'vertical')
plt.show()

#Statistical info about Price
data_1['Price'].describe()

#visual representation of the max, min, mean, outliers
fig = px.box(data_1.sort_values('Price', ascending=False),
             x="Airline",y="Price",orientation="v", title="Interactive Box Plot of Airline Prices")
fig.update_layout(xaxis_title="Airline", xaxis_tickangle=-45)
fig.show()

# In[23]:Feature Encoding
#use of one hot encoding for nominal data
for sub_category in data_1['Source'].unique():
    data_1['Source_'+sub_category] = data_1['Source'].apply(lambda x : 1 if x == sub_category else 0)
data_1.head(3)


# In[24]:
data_copy = data_1.copy()
#Target guided encoding
airline = data_copy.groupby(['Airline'])['Price'].mean().sort_values().index
air_dict = {key:index for index,key in enumerate(airline, 0)}
data_copy['Airline'] = data_copy['Airline'].map(air_dict)

#Target Guided ENcoding for destination
data_copy['Destination'].replace('New Delhi','Delhi', inplace = True)
data_copy['Destination'].unique()

dest = data_copy.groupby(['Destination'])['Price'].mean().sort_values().index
dest_dict = {key:index for index,key in enumerate(dest,0)}
data_copy['Destination'] = data_copy['Destination'].map(dest_dict)

# In[35]:
#LAbel encoding for ordinal data
data_copy["Total_Stops"].unique()
stops = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
data_copy['Total_Stops'] = data_copy['Total_Stops'].map(stops)

#Removing the useless columns
data_copy.drop(columns=['Additional_Info','Date_of_Journey','Route'],inplace = True)


# In[40]:
#handling outlier using IQR and replacing with median value
Q1 = data_copy['Price'].quantile(0.25)
Q3 = data_copy['Price'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

Max = Q3 + 1.5 * IQR
Min = Q3 - 1.5 * IQR

print('MAX:',Max, 'MIN:',Min)


# In[43]:
#finding outliers
print([price for price in data_copy['Price'] if price >= 23017 or price <= 1729])

#replacing outliers with median
data_copy["Price"] = np.where(data_copy['Price']>30000, data["Price"].median(),data['Price'])

# In[46]:
#feature selection before model building
from sklearn.feature_selection import mutual_info_regression

y = data_copy['Price']
X = data_copy.drop(['Price','Source','Total_duration(min)','Dep_Time','Arrival_Time','Duration'], axis=1)

imp = mutual_info_regression(X,y)
imp_df = pd.DataFrame(imp, index=X.columns) 
imp_df


# In[48]:
#building ml model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75, random_state=42)

# In[49]:
from sklearn.ensemble import RandomForestRegressor
ml_model = RandomForestRegressor()
ml_model.fit(X_train,y_train)

# In[50]:
y_pred = ml_model.predict(X_test)
ml_df = pd.DataFrame({'Y_test': y_test, 'Y_Pred': y_pred})
ml_df.head(3)

# In[69]:
#measuring accuracy
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test,y_pred)

# In[52]:
#saving ml model
import pickle
file = open(r'D:\data analysis\project\Ticket fare\model_ticket.pkl','wb')
pickle.dump(ml_model,file)

model = open(r'D:\data analysis\project\Ticket fare\model_ticket.pkl','rb')
forest = pickle.load(model)

#prediction based on X_train using the saved model - ml_model
y_pred2 = forest.predict(X_test)
model_df1 = pd.DataFrame({'Y_Test':y_test, 'Y_Pred':y_pred2})
model_df1.head(3)

#checking the accuracy 
round(r2_score(y_test,y_pred2),4)

#defining own evaluation metric
def mape(y_true, y_pred):  #mean abs perc error
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

mape(y_test,y_pred)


# In[63]:
#automating ML pipeline
def ml_pipeline(ml_model):
    model = ml_model.fit(X_train,y_train)                            #fitting the model
    print('Trainig Score: {}'.format(model.score(X_train,y_train)))  
    y_prediction = model.predict(X_test)                             #making prediction
    print('Prediction are:{}'.format(y_prediction))
    print('R2-Score:{}'.format(r2_score(y_test,y_prediction)))       #testing accuracy
    print('MSE:{}'.format(mean_squared_error(y_test,y_prediction)))
    print('MAPE:{}'.format(mape(y_test,y_prediction)))
    sns.distplot(y_test - y_prediction)                              #error on graph


# In[71]:
from sklearn.tree import DecisionTreeRegressor
ml_pipeline(DecisionTreeRegressor())

# In[72]:
ml_pipeline(RandomForestRegressor())
