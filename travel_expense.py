import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


flight = pd.read_csv(r"flights.csv")
users = pd.read_csv(r"users.csv")
hotel = pd.read_csv(r"hotels.csv")

flight.info()
flight.isna().sum()
flight.dtypes

users.info()
users.isna().sum()
users.dtypes

hotel.info()
hotel.isna().sum()
hotel.dtypes

users = users.rename(columns={'code': 'userCode'})
flight = flight.rename(columns={'price': 'flight_price'})
flight = flight.rename(columns={'from': 'from_loc'})
flight = flight.rename(columns={'to': 'to_loc'})
hotel = hotel.rename(columns={'price': 'hotel_price'})
hotel = hotel.rename(columns={'total': 'h_price'})

# Merge users and flight DataFrames based on 'user code'
merged_data = pd.merge(users, flight, on='userCode')

travel_data = pd.merge(merged_data, hotel, on=['userCode', 'travelCode'])

travel_data.isna().sum()
travel_data.duplicated().sum()

travel_data['total_expense'] = travel_data['flight_price'] + travel_data['h_price']

columns_to_drop = ['userCode', 'company', 'name_x', 'travelCode', 'time', 'date_x', 'name_y', 'h_price', 'date_y', 'agency']
travel = travel_data.drop(columns=columns_to_drop)


travel.columns

# Count the number of occurrences for each combination of 'from' and 'to' locations
travel_counts = travel.groupby(['from_loc', 'to_loc']).size().reset_index(name='count')

# Plot the travel counts
plt.figure(figsize=(10, 6))
plt.bar(travel_counts.index, travel_counts['count'])
plt.xlabel('Travel Combination Index')
plt.ylabel('Number of People')
plt.title('Number of People Who Traveled from One Location to Another')
plt.xticks(travel_counts.index, travel_counts['from_loc'] + ' to ' + travel_counts['to_loc'], rotation=90)
plt.tight_layout()
plt.show()


plt.hist(travel.hotel_price)
plt.title('Hotel Price')

plt.hist(travel.flight_price)
plt.title('Flight Price')


travel['hotel_price'].max()
travel['hotel_price'].min()

travel['flight_price'].max()
travel['flight_price'].min()

travel['total_expense'].max()
travel['total_expense'].min()

travel.dtypes

column_to_encode = ['gender', 'from_loc', 'to_loc', 'flightType', 'place']


for column in column_to_encode:
    label_encoder = LabelEncoder()
    travel[column] = label_encoder.fit_transform(travel[column])

travel_expense = travel['total_expense']


plt.boxplot(travel_expense)

# Add labels and title
plt.xlabel('Total Expense')
plt.ylabel('Value')
plt.title('Boxplot of Total Expense')

# Display the plot
plt.show()

travel.columns

scaler = StandardScaler()

total_expense_standardized = scaler.fit_transform(travel_expense.values.reshape(-1, 1))
scaled_flight_price = scaler.fit_transform(travel.flight_price.values.reshape(-1, 1)) 
scaled_hotel_price = scaler.fit_transform(travel.hotel_price.values.reshape(-1, 1))

travel['hotel_price'] = scaled_hotel_price
travel['flight_price'] = scaled_flight_price
travel['total_expense'] = total_expense_standardized

import seaborn as sns
sns.pairplot(travel.iloc[:, :])
plt.title('PairPlot')

corr = travel.corr()   

from scipy import stats
import pylab
stats.probplot(travel.total_expense, dist = "norm", plot = pylab)
plt.show()

travel.columns
travel.dtypes

travel['distance'] = travel['distance'].astype('int32')
travel['age'] = travel['age'].astype('int32')
travel['flight_price'] = travel['flight_price'].astype('int32')
travel['days'] = travel['days'].astype('int32')
travel['hotel_price'] = travel['hotel_price'].astype('int32')
travel['total_expense'] = travel['total_expense'].astype('int32')

import statsmodels.formula.api as smf 

model = smf.ols('total_expense ~ days + distance ', data = travel).fit()
model.summary()

#sm.graphics.influence_plot(model)


pred = model.predict(travel)

# Q-Q plot
res = model.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
#stats.probplot(res, dist = "norm", plot = pylab)
#plt.show()

### Splitting the data into train and test data
from sklearn.model_selection import train_test_split

travel_train, travel_test = train_test_split(travel_data, test_size = 0.2) # 20% test data

# preparing the model on train data
model_train = smf.ols('total_expense ~ days + distance', data = travel_train).fit()

# prediction on test data set
test_pred = model_train.predict(travel_test)
test_resid = test_pred - travel_test.total_expense
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

#rmse = 0.4127645176671771

# train_data prediction
train_pred = model_train.predict(travel_train)
train_resid  = train_pred - travel_train.total_expense
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

import pickle
filename = r'C:\Users\chink\Desktop\6th semester\AML\CA2\travel_expense_model.pkl'
pickle.dump(model_train, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

