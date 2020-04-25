import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
def readdata(date):
    data = pd.DataFrame()
    for _ in range(92):
        new_data = pd.read_csv(date.strftime("%Y-%m-%d")+".csv")
        date+=datetime.timedelta(days=1)
        data = pd.concat([data,new_data],ignore_index=True)
    return data
data = readdata(datetime.date(2019,10,1))
def conv_to_day(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').strftime("%A")
def conv_to_month(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').strftime("%B")
data['weekday'] = data.date.apply(lambda x: conv_to_day(x))
data['month'] = data.date.apply(lambda x: conv_to_month(x))
print(len(data.model.unique()))
print(data.shape)
print(data.head(5))
print(data.columns)

# plots of dates and failures
date_failure = data.groupby('date',as_index = False).agg({'failure':'sum'})
print(date_failure)
date_failure.plot(kind='line',x='date',y='failure',color = 'blue')
plt.show()

# Weekday to failure
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
category_day = pd.api.types.CategoricalDtype(categories=days, ordered=True)
wd_failure = data[['weekday', 'failure']].copy()
wd_failure['weekday'] = wd_failure['weekday'].astype(category_day)
wd_failure = wd_failure.groupby('weekday',as_index = False).agg({'failure':'sum'})
print(wd_failure.head(3))
wd_failure.plot(kind='bar',x='weekday',y='failure',color = 'grey')
plt.show()

# Month to Failure
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August','September','October','November','December']
category_month = pd.api.types.CategoricalDtype(categories=months, ordered=True)
month_failure = data[['month', 'failure']].copy()
month_failure['month'] = month_failure['month'].astype(category_month)
month_failure = month_failure.groupby('month',as_index = False).agg({'failure':'sum'})
month_failure = month_failure[month_failure.month.isin(['October','November','December'])]
print(month_failure.head(3))
month_failure.plot(kind='bar',x='month',y='failure',color = 'grey')
plt.show()

# Model to failure
model_failure = data.groupby('model',as_index = False).agg({'failure':'sum'})
print(model_failure)
model_failure.plot(kind='bar',x='model',y='failure',color='grey')
plt.show()
plt.savefig('model_to_failure.png')

# SMART correlation with Failure
columns = list(data)
columns = columns[5:-2]
smart_columns = []
for elem in columns:
    if 'normalized' in elem:
        smart_columns.append(elem)
print(smart_columns)
correlations = data[smart_columns].corrwith(data['failure'])
print(type(correlations))
correlations.dropna(inplace=True)
print(correlations)
correlations.plot(kind='bar',x='model',y='failure',color='grey')
plt.show()
plt.savefig('smart_correlation_to_failure.png')
sns.heatmap(correlations.to_frame(),cmap='RdYlGn')
plt.show()
plt.savefig('smart_correlation_to_failure_heatmap.png')