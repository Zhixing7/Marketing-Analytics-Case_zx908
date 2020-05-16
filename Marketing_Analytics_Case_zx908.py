#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from scipy import stats


# ## Import datasets

# #### 1.1 Subscibers dataset

# In[118]:


customer = pd.read_pickle(r'subscribers')

customer


# In[4]:


len(customer['subid'].unique())


# #### 1.2 Service dataset

# In[5]:


service = pd.read_pickle(r'customer_service_reps')

service


# In[10]:


for i in service.columns:
    print(i)
    print(sum(service[i].isnull()))


# In[36]:


service_by_individual = pd.pivot_table(service, index='subid', values=['current_sub_TF', 'num_trial_days', 
                            'trial_completed_TF', 'revenue_net_1month', 'payment_period'],
                                aggfunc = {'current_sub_TF':np.mean, 'num_trial_days':np.mean,
                                          'trial_completed_TF':np.mean, 'revenue_net_1month':np.mean,
                                          'payment_period':len})

service_by_individual.reset_index(drop=False, inplace=True)

service_by_individual.columns = ['subid', 'current_sub_TF', 'num_trial_days', 'total_payment_periods',
       'revenue_net_1month', 'trial_completed_TF']

service_by_individual


# In[37]:


service_by_individual['current_sub_TF'].value_counts()


# In[38]:


service_by_individual['num_trial_days'].value_counts()


# In[39]:


service_by_individual['trial_completed_TF'].value_counts()


# #### 1.3 Engagement dataset

# In[7]:


engag = pd.read_pickle(r'engagement')

engag


# In[8]:


len(engag['subid'].unique())


# In[19]:


feature_list = ['app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']

engag_by_individual = pd.pivot_table(engag, values=feature_list, index='subid', aggfunc=np.mean)
engag_by_individual.reset_index(drop=False, inplace=True)
engag_by_individual


# In[20]:


engag_at_period_0 = engag[engag['payment_period']==0]

print(engag_at_period_0.shape)

len(engag_at_period_0['subid'].unique())


# In[21]:


engag_period_0_by_individual = pd.pivot_table(engag_at_period_0, values=feature_list, index='subid', aggfunc=np.mean)
engag_period_0_by_individual.reset_index(drop=False, inplace=True)
engag_period_0_by_individual


# #### 1.4 Dataset-Combination for churn modeling (Only useful features from three dataset)

# In[22]:


customer.columns


# In[60]:


meaningful_features=['subid','package_type', 'preferred_genre', 'intended_use', 'plan_type',
                     'weekly_consumption_hour', 'age', 'male_TF', 'attribution_technical', 
                    'attribution_survey', 'cancel_before_trial_end', 'revenue_net', 'join_fee']

customer_with_meaningful_feature = customer[meaningful_features]

for i in meaningful_features:
    print(i)
    print(sum(customer_with_meaningful_feature[i].isnull()))


# In[61]:


#fill null category features with 'unknown'
customer_with_meaningful_feature['package_type'] = customer_with_meaningful_feature['package_type'].fillna('Unknown')
customer_with_meaningful_feature['preferred_genre'] = customer_with_meaningful_feature['preferred_genre'].fillna('Unknown')
customer_with_meaningful_feature['intended_use'] = customer_with_meaningful_feature['intended_use'].fillna('Unknown')
customer_with_meaningful_feature['male_TF'] = customer_with_meaningful_feature['male_TF'].fillna('Unknown')
customer_with_meaningful_feature['attribution_survey'] = customer_with_meaningful_feature['attribution_survey'].fillna('Unknown')

#fill null numeric features with 0
customer_with_meaningful_feature['revenue_net'] = customer_with_meaningful_feature['revenue_net'].fillna(0)
customer_with_meaningful_feature['join_fee'] = customer_with_meaningful_feature['join_fee'].fillna(0)

#fill unknown age and weekly_consumption_hour with 0 and add_a_remark_column
customer_with_meaningful_feature['age_null'] = customer_with_meaningful_feature['revenue_net'].isnull()
customer_with_meaningful_feature['consumption_hour_null'] = customer_with_meaningful_feature['weekly_consumption_hour'].isnull()

customer_with_meaningful_feature['age'] = customer_with_meaningful_feature['age'].fillna(0)
customer_with_meaningful_feature['weekly_consumption_hour'] = customer_with_meaningful_feature['weekly_consumption_hour'].fillna(0)

customer_with_meaningful_feature


# In[62]:


for i in meaningful_features:
    print(i)
    print(sum(customer_with_meaningful_feature[i].isnull()))


# In[250]:


#dataset.merge

modeling_dataset = pd.merge(customer_with_meaningful_feature, service_by_individual, on='subid', how='left')
modeling_dataset = pd.merge(modeling_dataset, engag_period_0_by_individual, on='subid', how='left')

modeling_dataset.shape


# In[251]:


modeling_dataset = modeling_dataset.dropna(axis=0)

modeling_dataset.shape


# In[252]:


modeling_dataset['current_sub_TF'].value_counts()


# ##### comparted two features indicate if customer finished free_trial

# In[74]:


modeling_dataset[['cancel_before_trial_end', 'trial_completed_TF']].head()


# In[73]:


sum(modeling_dataset['cancel_before_trial_end'] == modeling_dataset['trial_completed_TF'])


# In[253]:


modeling_dataset.drop(['cancel_before_trial_end'], axis=1, inplace=True)


# In[255]:


modeling_dataset_after_trial = modeling_dataset[modeling_dataset['trial_completed_TF']==True]

modeling_dataset_after_trial.shape


# ### Task 1: A-B Testing

# In[94]:


def t_test_1(A,B, alpha):
    
    x1 = np.mean(A)
    x2 = np.mean(B)
    p = (np.sum(A)+np.sum(B))/(len(A)+len(B))
    
    t = (x2-x1)/(np.sqrt(p*(1-p)*(1/len(A)+1/len(B))))
    
    critical_value = stats.t.ppf((1-alpha),(len(A)+len(B)-2))
    print(t)
    if np.abs(t) >= critical_value:
        print('reject null hypothesis with confidence interval of {}'.format(1-alpha))
    else:
        print('not reject null hypothesis with confidence interval of {}'.format(1-alpha))


# In[83]:


customer['plan_type'].value_counts()


# In[84]:


list_A = list(customer.loc[customer['plan_type']=='base_uae_14_day_trial', 'cancel_before_trial_end'])

list_B = list(customer.loc[customer['plan_type']=='low_uae_no_trial', 'cancel_before_trial_end'])

list_C = list(customer.loc[customer['plan_type']=='high_uae_14_day_trial', 'cancel_before_trial_end'])


# In[95]:


t_test_1(list_A, list_B, 0.01)


# In[98]:


t_test_1(list_B, list_C, 0.01)


# In[96]:


t_test_1(list_A, list_C, 0.05)


# ### Task 2: Customer Segmentation

# In[294]:


customer_for_cluster = customer[['preferred_genre',  'male_TF', 'age']]

customer_for_cluster = customer_for_cluster[customer_for_cluster['age']<=100]


customer_dummy = pd.get_dummies(customer_for_cluster)

customer_dummy


# In[284]:


customer_seg = pd.merge(customer[['subid']], customer_dummy, left_index=True, right_index = True, how='left')

customer_seg.shape


# In[285]:


customer_seg = pd.merge(customer_seg, engag_by_individual, on= 'subid', how='left')

customer_seg.shape


# In[286]:


customer_seg.dropna(axis=0, inplace=True)
#customer_seg = customer_seg[customer_seg['age']<=100]

customer_seg.set_index('subid',inplace=True)

customer_seg.shape


# In[106]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fitting(df):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return 


# In[289]:


fitting(customer_seg)


# In[290]:


kmeans = KMeans(n_clusters = 4, random_state=0).fit(customer_seg)


# In[291]:


a = list(kmeans.cluster_centers_)

seg_result = pd.DataFrame(a, columns=customer_seg.columns)

seg_result


# In[295]:


customer_dummy.dropna(axis=0, inplace=True)
customer_dummy

fitting(customer_dummy)


# In[297]:


kmeans = KMeans(n_clusters = 3, random_state=0).fit(customer_dummy)

a = list(kmeans.cluster_centers_)

seg_result = pd.DataFrame(a, columns=customer_dummy.columns)

seg_result


# ### Task 3: Churn Modeling

# In[256]:


modeling_dataset_after_trial.head()


# In[257]:


modeling = modeling_dataset.copy()

dummy_list = ['package_type','preferred_genre','male_TF','intended_use', 'plan_type', 'attribution_technical', 'attribution_survey']

model_dummy = pd.get_dummies(modeling[dummy_list])

model_dummy.shape


# In[258]:


modeling = modeling.drop(dummy_list, axis=1)

modeling = pd.merge(modeling, model_dummy, left_index=True, right_index = True, how='left')

modeling.shape


# In[213]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# In[266]:


#split data into training (70%) and test (30%) data set
X_dataset = modeling.drop(['current_sub_TF','subid','revenue_net_1month','total_payment_periods', 'trial_completed_TF','revenue_net'], axis=1)
y_data = modeling['current_sub_TF']
y_data=y_data.astype('int')

scaler = StandardScaler()

scaler.fit(X_dataset)

# Apply the power transform to the data
normal_X = pd.DataFrame(scaler.transform(X_dataset), columns = X_dataset.columns,
                              index = X_dataset.index)


X_train, X_test, y_train, y_test = train_test_split(normal_X, y_data, random_state = 0, test_size = 0.3)


# In[274]:


# built logisticregression model with C = [0.001, 0.1, 10, 1000], using cross validation to evaluate the models for contacted dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

clf = LogisticRegression()

cv_scores = cross_val_score(clf, X_train, y_train, cv=3)  # parameter cv is number of folds you want to split

print('Cross-validation scores for contacted dataset (3-fold):', cv_scores)
print('Mean cross-validation score for contacted dataset (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))

param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(clf, X_train, y_train,
                                            param_name='C',
                                            param_range=param_range, cv=5)

print('Mean train cross-validation score (5-folds) for contacted dataset with C = [0.001, 0.1, 10, 1000]: {}'.format(np.mean(train_scores, axis=1)))
print('Mean test cross-validation score (5-folds) for contacted dataset with C = [0.001, 0.1, 10, 1000]: {}'.format(np.mean(test_scores, axis=1)))


# In[261]:


from sklearn import metrics
def contacted_data_test(model):
    model.fit(X_train, y_train)
    #print("The coefficient of each independent variable is {}".format(model.coef_))
    print("The Mean test cross-validation score (5-folds) for contacted dataset: {}".format(np.mean(cross_val_score(model, X_train, y_train, cv=5))))

    prediction = model.predict(X_test)
    prob_log = model.predict_proba(X_test)[:,1]
    test_score = accuracy_score(y_test, prediction)
    print("The accuracy score for contacted dataset's test set: {}".format(test_score))

    CM = confusion_matrix(y_test,prediction)
    tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
    recall = tp/(tp+fn) #沒抓到
    precision = tp/(tp+fp) #抓錯
    print("Confusion matrix for the test set: {}".format(CM))
    print("TN: {} / FP: {} / FN: {} / TP: {}".format(tn, fp, fn, tp))
    print("Recall: {:.4f} / Precision: {:.4f}".format(recall, precision))
    
    print('AUC:{}'.format(metrics.roc_auc_score(y_test,prob_log)))


# In[275]:


# use the best parameter to build the logistic regression model for contacted dataset, apply on the test data and get the accuracy score
clf = LogisticRegression(C=0.1)
contacted_data_test(clf)


# In[267]:


# built decision tree model with max_depth= [3,4,5,6,7], using cross validation to evaluate the models for contacted dataset

from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(random_state=13)

cv_scores_4 = cross_val_score(dct, X_train, y_train, cv=3)  # parameter cv is number of folds you want to split

print('Cross-validation scores for contacted dataset(3-fold):', cv_scores_4)
print('Mean cross-validation score for contacted dataset(3-fold): {:.3f}'
     .format(np.mean(cv_scores_4)))

param_range = [3,4,5,6,7]
train_scores_2, test_scores_2 = validation_curve(dct, X_train, y_train,
                                            param_name='max_depth',
                                            param_range=param_range, cv=5)

print('Mean train cross-validation score (5-folds) for contacted dataset with max_depth = [3,4,5,6,7]: {}'.format(np.mean(train_scores_2, axis=1)))
print('Mean test cross-validation score (5-folds) for contacted dataset with max_depth = [3,4,5,6,7]: {}'.format(np.mean(test_scores_2, axis=1)))


# In[268]:


# use the best parameter to build the decision tree model for contacted dataset, apply on the test data and get the accuracy score
dct = DecisionTreeClassifier(random_state=13, max_depth=5)
contacted_data_test(dct)


# In[269]:


feat_importances = pd.DataFrame(dct.feature_importances_, index=X_train.columns, 
                             columns=['importance']).sort_values('importance', ascending=False).head(10)

feat_importances


# In[271]:


# built randomforest model with max_depth= [3,4,5], n_estimators = [13, 17, 20], max_features = [8,10,12,14], using gridsearch to evaluate the models for contacted dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rdf = RandomForestClassifier(random_state=1)

param_grid = [
    # try 24 (4×3×2) combinations of hyperparameters
    {'n_estimators': [50, 70], 'max_features': [10, 15], 'max_depth': [4,5,6]},
    # then try 6 (2×3) combinations with bootstrap set as False
    #{'bootstrap': [False], 'n_estimators': [5, 10], 'max_features': [2, 3, 4]},
  ]

grid_search = GridSearchCV(rdf, param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print(grid_search.best_estimator_)

# display the test score for each parameter combination
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[273]:


# use the best parameter to build the decision tree model for contacted dataset, apply on the test data and get the accuracy score
rdf = RandomForestClassifier(random_state=1, max_depth= 6, max_features= 15, n_estimators= 50)
contacted_data_test(rdf)


# In[298]:


feat_importances = pd.DataFrame(rdf.feature_importances_, index=X_train.columns, 
                             columns=['importance']).sort_values('importance', ascending=False).head(10)

feat_importances


# In[301]:


params = {'figure.figsize': (9,6),
          "font.size" : 16,
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'legend.fontsize': 15}
plt.rcParams.update(params)


# In[302]:


width = 0.5
plt.figure(figsize=(10,6))
plt.barh(feat_importances.index, feat_importances['importance'], color='#af4345')
plt.gca().invert_yaxis()
plt.show()

