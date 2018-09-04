# Ecommerce Traffic Segmentation
## Unsupervised Learning – Clustering

<img src= 'images/merch_store.png'>

### Motivation
Marketing is about reaching the right people, at the right time, in the right place. It's a never-ending process of learning more about potential customers and tailoring how and when you reach them. Understanding how users find your site, when they visit, and what device they use can help optimize media planning and increase ROI.

### Question
Can clustering be used to segment website traffic into useful personas? If so, how can the clusters inform marketing decisions?  

### Data
Data for this project came from the [Google Merchandise online store](https://shop.googlemerchandisestore.com/) and was retrieved from [Google Big Query](https://cloud.google.com/bigquery/). One year of data (>200,000 web sessions) is publicly available as a sample dataset for potential Google Big Query customers to explore. I used SQL to query 4 months of website sessions, where one row represents one session.

<img src= 'images/big_query.png'>


### Features
* Number of site visits – new or returning visitor
* Device – desktop or mobile
* Traffic medium – organic, paid, referral or affiliate
* Social referral - driven from social media or not
* Time of day – morning, afternoon, evening or night

### EDA
<img src= 'images/EDA_query.png' width="800">

<img src= 'images/EDA_medium.png' width="350"><img src= 'images/EDA_dtof.png' width="350" >


### Clustering
Sklearn's Kprototypes is a combination of KMeans and KModes. It's used for clustering a mix of numeric and categorical features. When fitting, KPrototypes takes the categorical indicies as an argument and clusters accordingly. KModes was another option, however initial results were not as interpretable.

```python
categorical_indicies = [1,2,3,4,5,6,7,8,9]

def get_cluster_labels(k):
  kproto = kprototypes.KPrototypes(n_clusters=k, init='Huang', max_iter=100)
  labels = kproto.fit_predict(X_scaled, categorical=categorical_indicies)
  return labels
```


### Ideal # of Clusters
Several methods were used to inform the ideal number of clusters. Although there was no clear elbow, and silhouette score continues to increase with k, I found points of interest at K=8 and K=9. PCA was used to visualize the data with cluster labels assigned to see clusters by color. Ultimately, I concluded 9 was the optimal number of clusters.

##### Elbow Curve
<img src= 'images/thursday_elbow_plot.png' width="800">

##### Principle Component Analysis
<img src = 'images/8_3D_PCA.png' width="400" height="350"> <img src= 'images/9_3D_PCA.png' width="400" height="350">

##### Silhouette Score
<img src= 'images/thurs_silouette_score.png' width="400">


### Clusters
To interpret clusters by feature, I labels to the original feature matrix, grouped by cluster, and aggregated by the mean.

| |   Number of  Visits |   Social Referral |   Device |   Organic |   Paid |   Referral & Affiliate |   Night |   Morning |   Afternoon |   Evening |
|----------:|-------------:|------------------:|---------:|----------:|-------:|---------------------:|--------:|----------:|------------:|----------:|
|         0 |         1.99 |              0.02 |     0.73 |      0.1  |   0.85 |                 0.05 |    0.23 |      0.24 |        0.2  |      0.22 |
|         1 |         1.48 |              0    |     0.37 |      0.95 |   0.05 |                 0    |    0.98 |      0.01 |        0    |      0    |
|         2 |         2.1  |              0.05 |     0.02 |      0    |   0.01 |                 0.99 |    0    |      0.06 |        0.83 |      0    |
|         3 |         1.37 |              0    |     0.28 |      0.94 |   0.06 |                 0    |    0    |      0.11 |        0    |      0.79 |
|         4 |         2.58 |              0.1  |     0.05 |      0    |   0.01 |                 0.99 |    0.9  |      0.03 |        0    |      0    |
|         5 |       157.64 |              0    |     0    |      0.4  |   0    |                 0.6  |    0.2  |      0    |        0.6  |      0.2  |
|         6 |        26.7  |              0.03 |     0.03 |      0.25 |   0.03 |                 0.72 |    0.18 |      0.03 |        0.33 |      0.39 |
|         7 |         1.7  |              0    |     0.24 |      0.95 |   0.05 |                 0    |    0    |      0.02 |        0.95 |      0    |
|         8 |         2.29 |              0.04 |     0.02 |      0    |   0.01 |                 0.99 |    0    |      0    |        0    |      1    |


### Interpretation
Below is a qualitative interpretation of cluster features based on the mean value of each column. Also added is column for cluster 'name' and size.

| Cluster Name | Size  |Number of Visits | Social Referral | Device | Traffic Medium | Time of Day |
| -- |:----:| ---:| ---:|----: |:----:| ---:|
|Dynamic Mobile Ad Clickers | 324 | Second Visit | No | Mobile | Paid | All day |
|New Insomniac Mobile Searchers | 948 | New Visitor | No | Some Mobile | Organic | Night |
|Work-Break Blog Readers | 1268 | Second Visit | No | Desktop | Referral/Affiliate | Afternoon |
|New After-Work Mobile Searchers | 1461 | New Visitor |  No | Some Mobile | Organic | Evening |
|Insomniac Social Blog Readers | 780 | 2-3 Visits | Some Social Referral  | Desktop | Referral/Affiliate | Night
|Indecisive Work-Break Browsers| 5 | Very Frequent Visitor | No | Desktop | Organic & Referral/Affiliate | Mostly Afternoon, Not AM |
|Low Funnel Work-Break Browsers| 72 | Frequent Visitor | No | Desktop | Referral/Affiliate | Mostly Afternoon, Not AM |
|Work-Break Mobile Searchers| 938 | Second Visit | No  | Some Mobile | Organic | Afternoon |
|After-Work Blog Readers| 1452 |Second Visit | No  | Desktop | Referral/Affiliate | Evening |

##### Insights
* new visitors are coming through mobile search in the evening and at night
* paid traffic is an all-day influx of returning users that leans mobile
* frequent visitors are coming through desktop mostly in the afternoon
* social sites are referring through desktop in the evening

### Cluster Performance
To see how each cluster performed on the site, I created a 'Y' table from the original query and added some custom metrics.

```python
Y['conversion_rate'] = Y['num_transactions'] / Y['cluster_size']
Y['conversion_value'] = Y['revenue'] / Y['num_transactions']
```

| Cluster Name                        |   Size |   Page Views |   Time on Site |   Action Type |   Revenue |  Conversion Rate |   Conversion Value |
|:------------------------------------|-------:|-------------:|---------------:|--------------:|----------:|------------------:|-------------------:|
| After-Work Blog Readers             |   1452 |        14.4  |           8.88 |          2.91 |  53081.4  |              0.28 |             132.15 |
| Dynamic Mobile Ad Clickers          |    324 |        12.17 |           7.8  |          1.82 |   1951.52 |              0.08 |              78.06 |
| Low Funnel Work-Break Browsers      |     72 |        16.27 |          12.32 |          3.08 |   7813.41 |              0.31 |             355.15 |
| Overnight Social Blog Readers       |    780 |        14.33 |           8.73 |          2.75 |  16243.4  |              0.24 |              85.49 |
| New After-Work Mobile Searchers     |   1461 |        12.6  |           8.34 |          1.92 |  15215.2  |              0.09 |             109.73 |
| New Overnight Mobile Searchers      |    948 |        12.82 |           7.57 |          1.91 |   5056.15 |              0.08 |              65.66 |
| Bottom Funnel Work-Break Browsers |      5 |        15.28 |          15.2  |          3.4  |    114.58 |              0.4  |              57.29 |
| Work-Break Blog Readers             |   1268 |        14.41 |           9.08 |          2.87 |  51397.1  |              0.27 |             147.48 |
| Work-Break Mobile Searchers         |    938 |        13    |           9.01 |          2.2  |  11962    |              0.13 |              96.99 |


<img src= 'images/conversion_rate.png' width="700">
<img src= 'images/conversion_value.png' width="700">

##### Insights

* 25% of afternoon & evening 'Blog Readers' are converting and spending
  * consider sponsored blog posts and promoting in the afternoon
* Paid Traffic has a low conversion rate
  * optimize ads for mobile
* Social sites are referring high conversion value customers between midnight-6am
  * consider paid social ads on late-night schedule

### Future Work
* Feature engineering
* Experiment with Kmodes
* EDA - products purchased, location of users, day of week, seasonality
* Supervised learning - predict actions on site, conversion value
