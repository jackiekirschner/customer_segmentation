import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from kmodes import kmodes
from kmodes import kprototypes
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import decomposition, datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# read in data
dfWorking = pd.read_csv('../data/USsessions.csv')


'''Data Cleaning'''

# clean columns
dfWorking['time_on_site'] = dfWorking['time_on_site'] / 60
dfWorking['revenue'] = dfWorking['revenue'] / (10**6)
dfWorking['social_referral'] = dfWorking.social_referral.map(dict(Yes=1, No=0))
dfWorking['device'] = dfWorking.device.map(dict(desktop=0, mobile=1, tablet=1))
# dfWorking['new_visitor'] = dfWorking.num_visits.where(dfWorking.num_visits<=1.0, 0)
# dfWorking.new_visitor= dfWorking.new_visitor.astype(int)
dfWorking['time_of_visit'] = pd.to_datetime(dfWorking['time_of_visit'],unit='s')
dfWorking = dfWorking.fillna(0)
dfWorking = dfWorking.drop(columns=['date'])

# adjust time of visit to local time
PST = ['Pleasanton', 'Tigard', 'Westlake Village', 'Hayward', 'Redwood City', 'San Mateo', 'Santa Ana', 'Bellevue', 'Fremont', 'Palo      Alto', 'Cupertino', 'San Bruno', 'Sunnyvale', 'Milpitas', 'San Jose', 'Bothell', 'Irvine', 'Portland', 'Oakland',
'Berkeley', 'Fresno', 'South San Francisco', 'Anaheim', 'Vancouver', 'San Francisco', 'Mountain View', 'Los Angeles', 'Kirkland', 'Santa Clara', 'Salem', 'Seattle', 'San Diego', 'Sacramento', 'Bellingham', 'Lake Oswego']

CST = ['Richardson', 'Lewisville', 'North Richland Hills', 'Pryor', 'Carrollton', 'Shiocton', 'Oshkosh', 'Evanston', 'McAllen',
'St. Louis', 'Omaha','Kansas City','Eau Claire','Chicago', 'Austin', 'Dallas','San Antonio', 'Minneapolis', 'Nashville','Madison','University Park']

MST = ['Rexburg','Avon','Orem','Tempe','Phoenix','Salt Lake City','Boulder', 'Denver','Thornton','Pueblo']

EST = ['Goose Creek','Akron', 'Wellesley', 'Cincinnati', 'Chamblee', 'Lenoir', 'Jersey City', 'Boardman', 'Piscataway Township', 'Indianapolis', 'Charlottesville', 'Charlotte', 'Raleigh', 'Ashburn', 'Dahlonega', 'Ann Arbor',
'New York', 'Pittsburgh','Amsterdam', 'Toronto', 'Miami', 'Boston', 'Orlando', 'College Park', 'Washington', 'Columbus', 'Kalamazoo', 'Atlanta', 'Newark', 'Philadelphia', 'Tampa', 'Detroit', 'Louisville', 'Miami', 'Sandy Springs']

def offset(city):
    return 3600 * (-8 * (city in PST) - 7 * (city in MST) - 6 * (city in CST)  - 5 * (city in EST))

dfWorking['local_time'] = pd.to_datetime(dfWorking['city'].map(offset) + dfWorking['time_of_visit'], unit='s')

# create 'time of day' categories from 'local time'
dfWorking = dfWorking.assign(time_of_day=pd.cut(dfWorking.local_time.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))

# get dummies for 'time of day'
dftod = pd.get_dummies(dfWorking["time_of_day"])

# get dummies for 'traffic medium'
dfMediums = pd.get_dummies(dfWorking["traffic_medium"])
# combine two paid traffic columns
dfMediums['paid'] = dfMediums['cpc']+dfMediums['cpm']
dfMediums = dfMediums.drop(columns=['cpc', 'cpm'])
# combine two traffic link columns
dfMediums['referral_affiliate'] = dfMediums['referral']+dfMediums['affiliate']
dfMediums = dfMediums.drop(columns=['referral', 'affiliate'])

# features for clustering
X = dfWorking[['num_visits', 'social_referral','device']]

# join X and dummy features for final features matrix
dfSessions = pd.concat([X, dfMediums, dftod], axis=1)

# standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(dfSessions)

'''Visualize Feature Data'''

# PCA

# pca = decomposition.PCA(n_components=10, copy=True, whiten=False, svd_solver="auto",
#           tol=0.0, iterated_power="auto", random_state=None)
# sessionsPCA = pca.fit_transform(scaled)

# def scree_plot(pca):
#     vals = pca.explained_variance_ratio_
#     plt.figure(figsize=(8, 4))
#     cum_var = np.cumsum(vals)
#     ax = plt.subplot(111)
#
#     ax.plot(range(len(vals) + 1), np.insert(cum_var, 0, 0), color = 'r', marker = 'o')
#     ax.bar(range(len(vals)), vals, alpha = 0.8)
#
#     ax.axhline(0.9, color = 'g', linestyle = "--")
#     ax.set_xlabel("Principal Component")
#     ax.set_ylabel("Variance Explained (%)")
#
#     plt.title("Scree Plot for the Sessions Dataset")
#
# scree_plot(pca)
# plt.show()
#  7 components explains >90% variance

# pca = decomposition.PCA(n_components=2)
#
# # Turn the dummified df into two columns with PCA
# plot_columns = pca.fit_transform(dfSessions)
#
# # Plot based on the two dimensions, and shade by cluster label
# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# plt.scatter(plot_columns[:,0], plot_columns[:,1], cmap='Greens')
# plt.title('PCA (3 Components) Feature Visualization')
# plt.savefig('2D_PCA.png')
# plt.show()


'''Clustering'''

# k-prototypes (for mixed numeric and categorical features)
# init = 'Huang' # can be 'Cao', 'Huang' or 'random'
# n_clusters = 8
# max_iter = 100
# kproto = kprototypes.KPrototypes(n_clusters=n_clusters,init=init, max_iter=max_iter)
#
# # fit/predict
# categoricals_indicies = [1,2,3,4,5,6,7,8,9]
# kproto_labels = kproto.fit_predict(scaled,categorical=categoricals_indicies)


# Elbow plot - Kproto
# distorsions = []
# for k in range(10,18):
#     kprot = kprototypes.KPrototypes(n_clusters=k)
#     kprot.fit(scaled, categorical=categoricals_indicies)
#     distorsions.append(kprot.cost_)
# # fig = plt.figure(figsize=(15, 5))
# # plt.plot(range(10, 18), distorsions)
# # plt.grid(True)
# # plt.title('Elbow curve')
# # plt.savefig('20_elbow_plot.png')
# # plt.show()
# print(distortions)
#
# #Silouette Score - Kproto
# def kproto_silhouette_score(nclust):
#     kprot = kprototypes.KPrototypes(nclust)
#     labels = kprot.fit_predict(scaled,categorical=categoricals_indicies)
#     sil_avg = silhouette_score(scaled, labels)
#     return sil_avg
# sil_scores = [kproto_silhouette_score(i) for i in range(7,17)]
# # plt.plot(range(7,17), sil_scores)
# # plt.xlabel('K')
# # plt.ylabel('Silhouette Score')
# # plt.title('Silhouette Score vs K')
# # plt.savefig('17_silouette_score.png')
# # plt.show()
# print(sil_scores)

# looks like ideal k = 8


#Kmodes

# km = kmodes.KModes(n_clusters=7, init='Huang', n_init=5, verbose=0)
# clusters = km.fit_predict(x)
# dfSessions['clusters'] = clusters

# Elbow plot - Kmodes
# cluster_range = range(2, 13)
# cluster_errors = []
#
# for num_clusters in cluster_range:
#   kprot = kprototypes.KPrototypes(num_clusters)
#   categoricals_indicies = [1,2,3,4,5,6,7,8]
#   kprot.fit(scaled, categorical=categoricals_indicies)
#   cluster_errors.append(kprot.cost_)
#
# clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
# print(clusters_df[0:10])
#
#
# plt.figure(figsize=(12,6))
# plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# plt.grid(True)
# plt.title('Elbow curve')
# plt.savefig('FE_kmodes_elbow_plot.png')
# plt.show()


# Silouette score - Kmodes
def kmodes_silhouette_score(nclust):
    # x = dfSessions
    # km = kmodes.KModes(n_clusters=nclust, init='Huang', n_init=5, verbose=0)
    # clusters = km.fit_predict(x)

    kprot = kprototypes.KPrototypes(nclust)
    categoricals_indicies = [1,2,3,4,5,6,7,8]
    clusters = kprot.fit_predict(scaled, categorical=categoricals_indicies)
    sil_avg = silhouette_score(scaled, clusters)
    return sil_avg

sil_scores = [kmodes_silhouette_score(i) for i in range(2,11)]
plt.plot(range(2,11), sil_scores)
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.savefig('FE_kmodes_silouette_score.png')
plt.show()


# copy features datafrme, add column with predicted cluster labels
# dfcopy = dfSessions.copy(deep=True)
# dfcopy['kproto_cluster'] = kproto_labels
#
# # # interpret clusters by taking mean
# means = dfcopy.groupby(['kproto_cluster']).mean()
