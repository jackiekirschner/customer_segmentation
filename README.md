# Ecommerce Traffic Segmentation
## Unsupervised Learning – Clustering

<img src= 'images/merch_store.png'>

### Question
Can website sessions be clustered by how and when users arrive at the site? If so, how can clusters be used to inform marketing decisions?  

### Data
I used data from the Google Merchandise online store. https://shop.googlemerchandisestore.com/ Google Analytics data from August 1, 2016 - August 1, 2017 (>200,000 web sessions) is publicly available on Google Big Query. I used SQL to query 4 months of website sessions, where one row represents one session.

<img src= 'images/big_query.png'>
```sql
sql query here
```

### Features
* Number of site visits – new or returning visitor
* Device – desktop or mobile
* Traffic medium – organic, paid, referral or affiliate
* Time of day – morning, afternoon, evening or night

### Clustering
Sklearn's Kprototypes is a combination of KMeans and KModes. It's used for clustering a mix of numeric and categorical features. When fitting, KPrototypes takes the categorical indicies as an argument and clusters accordingly.

```python
def get_labels(k):
  init = 'Huang'
  n_clusters = 8
  max_iter = 100

  kproto = kprototypes.KPrototypes(n_clusters=k,init=init, max_iter=max_iter)

  categoricals_indicies = [1,2,3,4,5,6,7,8,9]
  labels = kproto.fit_predict(X_scaled, categorical=categoricals_indicies)
  return labels
```

### Picking Ideal # of Clusters

##### Elbow
<img src= 'images/thursday_elbow_plot.png' width="800">

##### Principle Component Analysis
<img src = 'images/8_3D_PCA.png' width="400" height="350">
<img src= 'images/9_3D_PCA.png' width="400" height="350">

##### Silouette
<img src= 'images/thurs_silouette_score.png' width="400">


### Cluster Features
To interpret feature information about each cluster, I added a labels column to the original features dataframe, grouped by cluster, and aggregated column-wise by the mean.

| Cluster | # of visits | social referral | device | organic | paid | referral/affiliate | night | morning | afternoon | evening |
| -- |:----:| ---:|----: |:----:| ---:|----: |:---:| ----:|:---:| ----:|
|1| 1.504 | 0.0  | 0.263 |1.0| 0.0 |0.0 | 0.0| 0.0| 0.0 | 1.0|
|2| 157.637  | 0.0 |0.0| 0.400 |0.0 | 0.600  |0.200  |0.0 | 0.600 | 0.200 |
|3| 26.387| 0.027| 0.041 | 0.257 | 0.041| 0.703 | 0.176 | 0.0270  | 0.338 | 0.392
|4| 1.506 |  0.0 | 0.342 |1.00 | 0.0| 0.0 | 0.413  |0.101 |  0.397 | 0.0
|5| 1.855 | 0.0  | 0.387 | 0.0 | 1.0  | 0.0 | 0.266 | 0.0840  | 0.248|  0.344
|6| 2.260 | 0.045| 0.019 | 0.0| 0.0|1.0 | 0.0 | 0.0 | 0.0 | 1.0
|7| 2.046 | 0.054| 0.031 | 0.0 | 0.0 |1.0 | 0.0 | 0.068  | 0.822 | 0.0
|8| 2.644 | 0.097| 0.0517 | 0.011 | 0.0 | 0.990 | 0.897  |0.028 |  0.0 | 0.0


### Interpretation
Below is a qualitative interpretation of cluster features based on the means of each column.

| Cluster | Size  |# of visits | social referral | device | traffic medium | time of day |
| -- |:----:| ---:| ---:|----: |:----:| ---:|
|1| 1079 |New Visitor | No | Some Mobile | Organic| Evening|
|2| 5 |Very Frequent Visitor | No | Desktop| Organic & Ref/Affil  | Afternoon |
|3| 74 |Frequent Visitor | No | Desktop | Organic & Ref/Affil | Afternoon & Evening |
|4| 2120 | New Visitor |  No | Some Mobile | Organic | Night & Afternoon |
|5| 488 |New Visitor | No  | Some Mobile | Paid | Heavy Evening
|6| 1434 |Second Visit | No | Desktop | Ref/Affil | Evening |
|7| 1271 |Second Visit | No | Desktop | Ref/Affil | Afternoon |
|8| 777 |2-3 Visits | Some  | Desktop | Ref/Affil | Night |

### Clustering Insights

### Cluster Performance Insights
To see how each cluster performed on the site, I created a 'Y' dataframe with some additional columns.

```python
Y['conversion_rate'] = Y['num_transactions'] / Y['num_visits']
Y['avg_conversion_value'] = Y['revenue'] / Y['num_transactions']
```
* paid traffic leans mobile
* new visitors lean mobile (except for paid traffic), return visitors lean desktop
*

### EDA

* afternoon browsers spend more time on site

### Future Work

### Conclusion
