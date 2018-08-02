import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from kmodes import kprototypes
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from numpy.random import seed

seed(5)

# read in data
dfWorking = pd.read_csv('../data/USsessions.csv')

'''Data Cleaning'''
# clean columns
dfWorking['time_on_site'] = dfWorking['time_on_site'] / 60
dfWorking['revenue'] = dfWorking['revenue'] / (10**6)
dfWorking['social_referral'] = dfWorking.social_referral.map(dict(Yes=1, No=0))
dfWorking['device'] = dfWorking.device.map(dict(desktop=0, mobile=1, tablet=1))
dfWorking['time_of_visit'] = pd.to_datetime(dfWorking['time_of_visit'],unit='s')
dfWorking = dfWorking.fillna(0)
dfWorking = dfWorking.drop(columns=['date'])

# adjust time of visit to local time
PST=['Pleasanton', 'Tigard', 'Westlake Village', 'Hayward', 'Redwood City', 'San Mateo', 'Santa Ana', 'Bellevue', 'Fremont', 'Palo Alto', 'Cupertino', 'San Bruno', 'Sunnyvale', 'Milpitas', 'San Jose', 'Bothell', 'Irvine', 'Portland', 'Oakland',
'Berkeley', 'Fresno', 'South San Francisco', 'Anaheim', 'Vancouver', 'San Francisco', 'Mountain View', 'Los Angeles', 'Kirkland', 'Santa Clara', 'Salem', 'Seattle', 'San Diego', 'Sacramento', 'Bellingham', 'Lake Oswego']

CST=['Richardson', 'Lewisville', 'North Richland Hills', 'Pryor', 'Carrollton', 'Shiocton', 'Oshkosh', 'Evanston', 'McAllen',
'St. Louis', 'Omaha','Kansas City','Eau Claire','Chicago', 'Austin', 'Dallas','San Antonio', 'Minneapolis', 'Nashville','Madison','University Park']

MST=['Rexburg','Avon','Orem','Tempe','Phoenix','Salt Lake City','Boulder', 'Denver','Thornton','Pueblo']

EST=['Goose Creek','Akron', 'Wellesley', 'Cincinnati', 'Chamblee', 'Lenoir', 'Jersey City', 'Boardman', 'Piscataway Township', 'Indianapolis', 'Charlottesville', 'Charlotte', 'Raleigh', 'Ashburn', 'Dahlonega', 'Ann Arbor',
'New York', 'Pittsburgh','Amsterdam', 'Toronto', 'Miami', 'Boston', 'Orlando', 'College Park', 'Washington', 'Columbus', 'Kalamazoo', 'Atlanta', 'Newark', 'Philadelphia', 'Tampa', 'Detroit', 'Louisville', 'Miami', 'Sandy Springs']

def offset(city):
    return 3600 * (-8 * (city in PST) - 7 * (city in MST) - 6 * (city in CST)  - 5 * (city in EST))

dfWorking['local_time'] = pd.to_datetime(dfWorking['city'].map(offset) + dfWorking['time_of_visit'], unit='s')

# create 'time of day' categorical column from local time
dfWorking = dfWorking.assign(time_of_day=pd.cut(dfWorking.local_time.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))

# features for clustering
X = dfWorking[['num_visits', 'social_referral', 'device']]

# get dummies for 'traffic medium'
dfMediums = pd.get_dummies(dfWorking["traffic_medium"])
#combine two paid traffic columns
dfMediums['paid'] = dfMediums['cpc']+dfMediums['cpm']
dfMediums = dfMediums.drop(columns=['cpc', 'cpm'])
# combine two traffic link columns
dfMediums['referral_affiliate'] = dfMediums['referral']+dfMediums['affiliate']
dfMediums = dfMediums.drop(columns=['referral', 'affiliate'])

# get dummies for time of day
dftod = pd.get_dummies(dfWorking["time_of_day"])

# join X and dummied features
dfSessions =  pd.concat([X, dfMediums, dftod], axis=1)


# standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(dfSessions)

'''Clustering'''
# k-prototypes (mixed numeric and categorical features)
init = 'Huang'
n_clusters = 5
max_iter = 100

kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init, max_iter=max_iter)

# fit/predict
categoricals_indicies = [1,2,3,4,5,6,7,8,9]
labels = kproto.fit_predict(scaled, categorical=categoricals_indicies)

# copy features datafrme, add column with predicted cluster labels
dfcopy = dfSessions.copy(deep=True)
dfcopy['cluster'] = labels

# interpret clusters by taking mean
means = dfcopy.groupby(['cluster']).mean()

#
dfY =  pd.concat([dfWorking, dfMediums, dftod], axis=1)
dfY.drop(columns=['user_id', 'time_of_visit', 'traffic_medium',
       'traffic_source', 'social_referral', 'device', 'local_time', 'time_of_day', 'organic', 'paid',
       'referral_affiliate', 'Night', 'Morning', 'Afternoon', 'Evening'], inplace=True)
dfY['cluster'] = labels
dfCounts = dfY.cluster.value_counts()

dfY = dfY.groupby('cluster').agg({'page_views':'mean','time_on_site':'mean', 'num_visits': 'sum',
'num_transactions':'sum','revenue': 'sum', 'action_type': 'mean'})

dfY['size'] = dfCounts.iloc[:]

dfY['conversion_rate'] = dfY['num_transactions'] / dfY['num_visits']
dfY['avg_conversion_value'] = dfY['revenue'] / dfY['num_transactions']

'''Ideal K'''

# Elbow plot
distorsions = []
for k in range(2, 10):
    kprot = kprototypes.KPrototypes(n_clusters=k)
    kprot.fit(scaled, categorical=categoricals_indicies)
    distorsions.append(kprot.cost_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 10), distorsions)
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Sum Distance to Centroids')
plt.title('Elbow curve')
# plt.savefig('wed_elbow_plot.png')
plt.show()

# Silouette Score
# def get_silhouette_score(nclust):
#     kprot = kprototypes.KPrototypes(nclust)
#     labels = kprot.fit_predict(scaled,categorical=categoricals_indicies)
#     sil_avg = silhouette_score(scaled, labels)
#     return sil_avg
#
# sil_scores = [get_silhouette_score(i) for i in range(2,11)]
# plt.plot(range(2,11), sil_scores)
# plt.grid(True)
# plt.xlabel('K')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs K')
# plt.savefig('wed_silouette_score.png')
# plt.show()



'''Data Visualiztion'''
dfcopy['cluster']

# 2D
# Transform df into two features with PCA
# pca = PCA(2)
# plot_columns = pca.fit_transform(scaled)
# # Plot based on two dimensions, and shade by cluster label
# plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=dfcopy['cluster'], s=30)
# plt.title('PCA (2 Components) Feature Visualization')
# plt.savefig('wed_2D_PCA.png')
# plt.show()

# 3D
# Transform df into three features with PCA
# pca = PCA(3)
# plot_columns = pca.fit_transform(scaled)
# # Plot based on three dimensions, and shade by cluster label
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(plot_columns[:,0], plot_columns[:,1], plot_columns[:,2], c=dfcopy['cluster'], s=30)
# ax.set_alpha(0.0001)
# plt.title('PCA (3 Components) Feature Visualization')
# plt.savefig('wed_3D_PCA.png')
# plt.show()
