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
from tabulate import tabulate

seed(5)

# read in data
dfQuery = pd.read_csv('../data/USsessions.csv')

'''Data Cleaning'''
# clean columns
dfQuery['time_on_site'] = dfQuery['time_on_site'] / 60
dfQuery['revenue'] = dfQuery['revenue'] / (10**6)
dfQuery['social_referral'] = dfQuery.social_referral.map(dict(Yes=1, No=0))
dfQuery['device'] = dfQuery.device.map(dict(desktop=0, mobile=1, tablet=1))
dfQuery['time_of_visit'] = pd.to_datetime(dfQuery['time_of_visit'],unit='s')
dfQuery = dfQuery.fillna(0)
dfQuery = dfQuery.drop(columns=['date'])

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

dfQuery['local_time'] = pd.to_datetime(dfQuery['city'].map(offset) + dfQuery['time_of_visit'], unit='s')

# create 'time of day' categorical column from local time
dfQuery = dfQuery.assign(time_of_day=pd.cut(dfQuery.local_time.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))


'''Features for Clustering'''
X = dfQuery[['num_visits', 'social_referral', 'device']]

# get dummies for 'traffic medium'
dfMediums = pd.get_dummies(dfQuery["traffic_medium"])
# combine two paid traffic columns
dfMediums['paid'] = dfMediums['cpc']+dfMediums['cpm']
# combine two traffic link columns
dfMediums['referral_affiliate'] = dfMediums['referral']+dfMediums['affiliate']
# drop now-combined columns
dfMediums = dfMediums.drop(columns=['referral', 'affiliate', 'cpc', 'cpm'])

# get dummies for time of day
dftod = pd.get_dummies(dfQuery["time_of_day"])

# join X and dummied features for final feature matrix
X =  pd.concat([X, dfMediums, dftod], axis=1)

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''Clustering'''
# k-prototypes (mixed numeric and categorical features)
init = 'Huang'
n_clusters = 9
max_iter = 100

kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init, max_iter=max_iter)

# fit/predict
categoricals_indicies = [1,2,3,4,5,6,7,8,9]
labels = kproto.fit_predict(X_scaled, categorical=categoricals_indicies)

# add column to original feature matrix with predicted cluster labels
X['cluster'] = labels

# interpret clusters by taking column-wise mean of features
means = X.groupby(['cluster']).mean()

'''Ideal K'''

# Elbow plot
# distorsions = []
# for k in range(2, 11):
#     kprot = kprototypes.KPrototypes(n_clusters=k)
#     kprot.fit(X_scaled, categorical=categoricals_indicies)
#     distorsions.append(kprot.cost_)
#
# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 11), distorsions)
# plt.grid(True)
# plt.xlabel('K')
# plt.ylabel('Sum Distance to Centroids')
# plt.title('Elbow curve')
# plt.savefig('../images/thursday_elbow_plot.png')
# plt.show()

# # Silouette Score
# def get_silhouette_score(nclust):
#     kprot = kprototypes.KPrototypes(nclust)
#     labels = kprot.fit_predict(X_scaled, categorical=categoricals_indicies)
#     sil_avg = silhouette_score(X_scaled, labels)
#     return sil_avg
# #
# sil_scores = [get_silhouette_score(i) for i in range(2,11)]
# plt.plot(range(2,11), sil_scores)
# plt.grid(True)
# plt.xlabel('K')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs K')
# plt.savefig('../images/thurs_silouette_score.png')
# plt.show()



'''Data Visualiztion'''

# # 2D
# # Transform df into two features with PCA
# pca = PCA(2)
# plot_columns = pca.fit_transform(X_scaled)
# # Plot based on two dimensions, and shade by cluster label
# fig, ax = plt.subplots()
# ax.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels,cmap='nipy_spectral', s=30)
# ax.set_alpha(0.0001)
# plt.title('K=8 Feature Visualization')
# plt.savefig('../images/9_2D_PCA.png')
# plt.show()

# # 3D
# # Transform df into three features with PCA
# pca = PCA(3)
# plot_columns = pca.fit_transform(X_scaled)
# # Plot based on three dimensions, and shade by cluster label
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(plot_columns[:,0], plot_columns[:,1], plot_columns[:,2], c=labels, cmap='nipy_spectral',s=30)
# ax.set_alpha(0.0001)
# plt.title('K=9 Feature Visualization')
# plt.savefig('../images/9_3D_PCA.png')
# plt.show()

'''Features EDA'''
# % mobile vs. Desktop
# %

'''EDA by Cluster'''
# create 'Y' table to see cluster performance on site
Y = dfQuery[['page_views', 'time_on_site', 'num_visits', 'num_transactions', 'revenue','action_type']]

Y['cluster'] = labels

# assign cluster names based on interpretations
cluster_names = ['Dynamic Mobile Ad Clickers','New Insomniac Mobile Searchers', 'Work-Break Blog Readers', 'New After-Work Mobile Searchers', 'Insomniac Social Blog Readers', 'Very Indecisive Work-Break Browsers', 'Indecisive Work-Break Browsers', 'Work-Break Mobile Searchers', 'After-Work Blog Readers']

Y['cluster_name'] = [cluster_names[i] for i in labels]


cluster_sizes = Y.cluster.value_counts()

Y = Y.groupby('cluster_name').agg({'page_views':'mean','time_on_site':'mean', 'num_visits': 'sum', 'num_transactions':'sum','revenue': 'sum', 'action_type': 'mean'})

# create new performace metric columns
Y['conversion_rate'] = Y['num_transactions'] / Y['num_visits']
Y['conversion_value'] = Y['revenue'] / Y['num_transactions']

Y['size'] = cluster_sizes.iloc[:]


# print(tabulate(means.round(2), headers='keys', tablefmt='pipe'))
# print(tabulate(Y.round(2), headers='keys', tablefmt='pipe'))

# plot conversion rate by cluster
cr = Y.xs('conversion_rate', axis=1)
plt.figure(figsize =(11,7))
ax1 = cr.plot(kind='bar', title='Conversion Rate')
plt.xticks(rotation = 20, fontsize = 6)
plt.savefig('../images/conversion_rate.png')
plt.show()


# conversion value by cluster
cv = Y.xs('conversion_value', axis=1)
plt.figure(figsize =(11,7))
cv.plot(kind='bar', title='Conversion Value')
plt.xticks(rotation = 20, fontsize = 6)
plt.savefig('../images/conversion_value.png')
plt.show()
