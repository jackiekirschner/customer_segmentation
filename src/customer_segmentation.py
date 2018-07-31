import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from kmodes import kmodes
from kmodes import kprototypes
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# read in data
dfWorking = pd.read_csv('../data/USsessions.csv')

# clean columns
dfWorking['time_on_site'] = dfWorking['time_on_site'] / 60
dfWorking['revenue'] = dfWorking['revenue'] / (10**6)
dfWorking['social_referral'] = dfWorking.social_referral.map(dict(Yes=1, No=0))
dfWorking['device'] = dfWorking.device.map(dict(desktop=0, mobile=1, tablet=1))
dfWorking = dfWorking.fillna(0)
dfWorking = dfWorking.drop(columns=['date'])

# adjust time of visit to local time
PST = ['Pleasanton', 'Tigard', 'Westlake Village', 'Hayward', 'Redwood City', 'San Mateo', 'Santa Ana', 'Bellevue', 'Fremont', 'Palo Alto', 'Cupertino', 'San Bruno', 'Sunnyvale', 'Milpitas', 'San Jose', 'Bothell', 'Irvine', 'Portland', 'Oakland',
'Berkeley', 'Fresno', 'South San Francisco', 'Anaheim', 'Vancouver', 'San Francisco', 'Mountain View', 'Los Angeles', 'Kirkland', 'Santa Clara', 'Salem', 'Seattle', 'San Diego', 'Sacramento', 'Bellingham', 'Lake Oswego']

CST = ['Richardson', 'Lewisville', 'North Richland Hills', 'Pryor', 'Carrollton', 'Shiocton', 'Oshkosh', 'Evanston', 'McAllen',
'St. Louis', 'Omaha','Kansas City','Eau Claire','Chicago', 'Austin', 'Dallas','San Antonio', 'Minneapolis', 'Nashville','Madison','University Park']

MST = ['Rexburg','Avon','Orem','Tempe','Phoenix','Salt Lake City','Boulder', 'Denver','Thornton','Pueblo']

EST = ['Goose Creek','Akron', 'Wellesley', 'Cincinnati', 'Chamblee', 'Lenoir', 'Jersey City', 'Boardman', 'Piscataway Township', 'Indianapolis', 'Charlottesville', 'Charlotte', 'Raleigh', 'Ashburn', 'Dahlonega', 'Ann Arbor',
'New York', 'Pittsburgh','Amsterdam', 'Toronto', 'Miami', 'Boston', 'Orlando', 'College Park', 'Washington', 'Columbus', 'Kalamazoo', 'Atlanta', 'Newark', 'Philadelphia', 'Tampa', 'Detroit', 'Louisville', 'Miami', 'Sandy Springs']

def offset(city):
    return 3600 * (-8 * (city in PST) - 7 * (city in MST) - 6 * (city in CST)  - 5 * (city in EST))

# POSIX to datetime
dfWorking['local_time'] = pd.to_datetime(dfWorking['city'].map(offset) + dfWorking['time_of_visit'], unit='s')
dfWorking['time_of_visit'] = pd.to_datetime(dfWorking['time_of_visit'],unit='s')

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

# normalize
scaler = StandardScaler()
scaled = scaler.fit_transform(dfSessions)

# clustering k-prototypes (mixed numeric and categorical features)

init = 'Huang' # can be 'Cao', 'Huang' or 'random'
n_clusters = 8
max_iter = 100

kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init, max_iter=max_iter)
# k_prototypes(X, categorical, n_clusters, max_iter, num_dissim,
# cat_dissim, gamma, init, n_init, verbose, random_state, n_jobs)

    # cluster_centroids_ : array, [n_clusters, n_features]
    #     Categories of cluster centroids
    # labels_ :
    #     Labels of each point
    # cost_ : float
    #     Clustering cost, defined as the sum distance of all points to
    #     their respective cluster centroids.
    # n_iter_ : int
    #     The number of iterations the algorithm ran for.
    # gamma : float
    #     The (potentially calculated) weighing factor.
# fit/predict
categoricals_indicies = [1,2,3,4,5,6,7,8,9]
labels = kproto.fit_predict(scaled,categorical=categoricals_indicies)


# Elbow plot
# distorsions = []
# for k in range(2, 10):
#     kprot = kprototypes.KPrototypes(n_clusters=k)
#     kprot.fit(scaled, categorical=categoricals_indicies)
#     distorsions.append(kprot.cost_)
#
# fig = plt.figure(figsize=(15, 5))
# plt.plot(range(2, 10), distorsions)
# plt.grid(True)
# plt.title('Elbow curve')
# plt.savefig('elbow_plot.png')
# plt.show()

# Silouette Score
# def get_silhouette_score(nclust):
#     kprot = kprototypes.KPrototypes(nclust)
#     labels = kprot.fit_predict(scaled,categorical=categoricals_indicies)
#     sil_avg = silhouette_score(scaled, labels)
#     return sil_avg

# looks like ideal k = 8


# sil_scores = [get_silhouette_score(i) for i in range(2,10)]
# plt.plot(range(2,10), sil_scores)
# plt.xlabel('K')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs K')
# plt.savefig('silouette_score.png')
# plt.show()


# copy features datafrme, add column with predicted cluster labels
dfcopy = dfSessions
dfcopy['cluster'] = labels

# interpret clusters by taking mean
means = dfcopy.groupby(['cluster']).mean()

# Scree plot
# Silouette Score
# Gap

#
#
#
#
#
#
#
# # ex 2
#
# kproto = KPrototypes(n_clusters=15, init='Cao', verbose=2)
# clusters = kproto.fit_predict(X, categorical=[1, 2])
# # Print cluster centroids of the trained model.
# print(kproto.cluster_centroids_)
# # Print training statistics
# print(kproto.cost_)
# print(kproto.n_iter_)
# for s, c in zip(syms, clusters):
#     print("Result: {}, cluster:{}".format(s, c))
# # Plot the results
# for i in set(kproto.labels_):
#     index = kproto.labels_ == i
#     plt.plot(X[index, 0], X[index, 1], 'o')
#     plt.suptitle('Data points categorized with category score', fontsize=18)
#     plt.xlabel('Category Score', fontsize=16)
#     plt.ylabel('Category Type', fontsize=16)
# plt.show()
# # Clustered result
# fig1, ax3 = plt.subplots()
# scatter = ax3.scatter(syms, clusters, c=clusters, s=50)
# ax3.set_xlabel('Data points')
# ax3.set_ylabel('Cluster')
# plt.colorbar(scatter)
# ax3.set_title('Data points classifed according to known centers')
# plt.show()
# result = zip(syms, kproto.labels_)
# sortedR = sorted(result, key=lambda x: x[1])
# print(sortedR)
#
#
# # silhouette score
# from sklearn.metrics import silhouette_score
# def get_silhouette_score(nclust):
#     km = KMeans(nclust)
#     km.fit(X)
#     sil_avg = silhouette_score(X, km.labels_)
#     return sil_avg
# sil_scores = [get_silhouette_score(i) for i in range(2,10)]
# plt.plot(range(2,10), sil_scores)
# plt.xlabel('K')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs K')
