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

# # adjust time of visit to local time
# # add unique cities to respective time zone
# print(dfQuery['city'].unique())

PST=['Pleasanton', 'Tigard', 'Westlake Village', 'Hayward', 'Redwood City', 'San Mateo', 'Santa Ana', 'Bellevue', 'Fremont', 'Palo Alto', 'Cupertino', 'San Bruno', 'Sunnyvale', 'Milpitas', 'San Jose', 'Bothell', 'Irvine', 'Portland', 'Oakland',
'Berkeley', 'Fresno', 'South San Francisco', 'Anaheim', 'Vancouver', 'San Francisco', 'Mountain View', 'Los Angeles', 'Kirkland', 'Santa Clara', 'Salem', 'Seattle', 'San Diego', 'Sacramento', 'Bellingham', 'Lake Oswego']
CST=['Richardson', 'Lewisville', 'North Richland Hills', 'Pryor', 'Carrollton', 'Shiocton', 'Oshkosh', 'Evanston', 'McAllen', 'St. Louis', 'Omaha','Kansas City','Eau Claire','Chicago', 'Austin', 'Dallas','San Antonio', 'Minneapolis', 'Nashville','Madison','University Park']
MST=['Rexburg','Avon','Orem','Tempe','Phoenix','Salt Lake City','Boulder', 'Denver','Thornton','Pueblo']
EST=['Goose Creek','Akron', 'Wellesley', 'Cincinnati', 'Chamblee', 'Lenoir', 'Jersey City', 'Boardman', 'Piscataway Township', 'Indianapolis', 'Charlottesville', 'Charlotte', 'Raleigh', 'Ashburn', 'Dahlonega', 'Ann Arbor',
'New York', 'Pittsburgh','Amsterdam', 'Toronto', 'Miami', 'Boston', 'Orlando', 'College Park', 'Washington', 'Columbus', 'Kalamazoo', 'Atlanta', 'Newark', 'Philadelphia', 'Tampa', 'Detroit', 'Louisville', 'Miami', 'Sandy Springs']

# create 'local time' column
def offset(city):
    return 3600 * (-8 * (city in PST) - 7 * (city in MST) - 6 * (city in CST)  - 5 * (city in EST))

dfQuery['local_time'] = pd.to_datetime(dfQuery['city'].map(offset) + dfQuery['time_of_visit'], unit='s')

# create categorial column 'time of day' from local time
dfQuery = dfQuery.assign(time_of_day=pd.cut(dfQuery.local_time.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))


'''Prep Feature Matrix for Clustering'''

# features that don't need dummies
X = dfQuery[['num_visits', 'social_referral', 'device']]

# dummies for 'traffic medium'
dfMediums = pd.get_dummies(dfQuery["traffic_medium"])
# combine similar dummies to reduce dimensions
dfMediums['paid'] = dfMediums['cpc']+dfMediums['cpm']
dfMediums['referral_affiliate'] = dfMediums['referral']+dfMediums['affiliate']
# drop extraneous colums
dfMediums = dfMediums.drop(columns=['cpc', 'cpm','referral', 'affiliate'])

# get dummies for time of day
dftod = pd.get_dummies(dfQuery["time_of_day"])

# join X and dummied features for final feature matrix
X = pd.concat([X, dfMediums, dftod], axis=1)

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''Clustering Method'''
# KPrototypes (mixed numeric and categorical features)
categorical_indicies = [1,2,3,4,5,6,7,8,9]
def get_cluster_labels(num_clusters):
    kproto = kprototypes.KPrototypes(n_clusters=num_clusters, init='Huang')
    labels = kproto.fit_predict(X_scaled, categorical=categorical_indicies)
    return labels

def get_cluster_cost(num_clusters):
    kproto = kprototypes.KPrototypes(n_clusters=num_clusters, init='Huang')
    kproto.fit(X_scaled, categorical=categorical_indicies)
    cost = kproto.cost_
    return cost

'''Picking Ideal K'''
# Elbow plot
def plot_elbow_curve(max_range):
    distorsions = []
    for i in range(2, max_range):
        cost = get_cluster_cost(i)
        distorsions.append(cost)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_range), distorsions)
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Sum Distance to Centroids')
    plt.title('Elbow curve')
    plt.savefig('../images/test_elbow_plot.png')
    plt.show()

# Silouette Score
def get_silhouette_score(num_clusters):
    labels = get_cluster_labels(num_clusters)
    sil_avg = silhouette_score(X_scaled, labels)
    return sil_avg

def plot_silhouette_score(max_range):
    sil_scores = [get_silhouette_score(i) for i in range(2, max_range)]
    plt.plot(range(2,max_range), sil_scores)
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs K')
    plt.savefig('../images/test_silouette_score.png')
    plt.show()

# PCA Data Visualization
def plot_PCA(num_clusters, num_components):
    pca = PCA(num_components)
    # num_clusters needed to visualize data points by predicted label (add color)
    labels = get_cluster_labels(num_clusters)
    plot_columns = pca.fit_transform(X_scaled)

    if num_components == 2:
        fig, ax = plt.subplots()
        ax.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels,cmap='nipy_spectral', s=30)
        ax.set_alpha(0.0001)
        plt.savefig('../test/test_2D_PCA.png')
        plt.show()
    elif num_components == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(plot_columns[:,0], plot_columns[:,1], plot_columns[:,2], c=labels, cmap='nipy_spectral',s=30)
        ax.set_alpha(0.0001)
        plt.savefig('../images/test_3D_PCA.png')

    plt.title('K={} Feature Visualization'.format(num_clusters))
    plt.show()


plot_elbow_curve(max_range=5)
plot_silhouette_score(max_range=5)
plot_PCA(num_clusters=9,num_components=3)
# conclude that ideal k = 9

# add cluster labels to original feature matrix
labels = get_cluster_labels(9)
X['cluster'] = labels

# group feature matrix by cluster and aggregate by mean to interpret clusters
cluster_feature_means = X.groupby(['cluster']).mean()



'''Query EDA'''
# Counts of Features being clustered
# % mobile vs. Desktop
# % tables print

'''EDA by Cluster'''
# create 'Y' table to see cluster performance on site

# remove from y: num_visits	num_transactions revenue
# change new columns to grab from other dataframe

# Y = dfQuery[['page_views', 'time_on_site', 'num_visits', 'num_transactions', 'revenue','action_type']]
# Y['cluster'] = labels
#
# # assign cluster names based on interpretations
# cluster_names = ['Dynamic Mobile Ad Clickers','New Insomniac Mobile Searchers', 'Work-Break Blog Readers', 'New After-Work Mobile Searchers', 'Insomniac Social Blog Readers', 'Very Indecisive Work-Break Browsers', 'Indecisive Work-Break Browsers', 'Work-Break Mobile Searchers', 'After-Work Blog Readers']
#
# Y['cluster_name'] = [cluster_names[i] for i in labels]
#
# cluster_sizes = Y.cluster_name.value_counts()
#
# Y = Y.groupby('cluster_name').agg({'page_views':'mean','time_on_site':'mean', 'num_visits': 'sum', 'num_transactions':'sum','revenue': 'sum', 'action_type': 'mean'})
#
# Y['size'] = cluster_sizes.iloc[:]
#
# # create new performace metric columns
# Y['conversion_rate'] = Y['num_transactions'] / Y['size']
# Y['conversion_value'] = Y['revenue'] / Y['num_transactions']
#
#
# # plot conversion rate by cluster
# cr = Y.xs('conversion_rate', axis=1)
# plt.figure(figsize =(11,7))
# ax1 = cr.plot(kind='bar', title='Conversion Rate')
# plt.xticks(rotation = 15, fontsize = 8)
# plt.grid(True)
# plt.savefig('../images/conversion_rate.png')
# plt.show()
#
#
# # conversion value by cluster
# cv = Y.xs('conversion_value', axis=1)
# plt.figure(figsize =(11,7))
# cv.plot(kind='bar', title='Conversion Value')
# plt.xticks(rotation = 15, fontsize = 8)
# plt.grid(True)
# plt.savefig('../images/conversion_value.png')
# plt.show()

# # print dataframe tables
# print(tabulate(means.round(2), headers='keys', tablefmt='pipe'))
# print(tabulate(Y.round(2), headers='keys', tablefmt='pipe'))
