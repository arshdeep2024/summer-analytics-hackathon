#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[187]:


training_set_features = pd.read_csv("training_set_features.csv")
training_set_labels = pd.read_csv("training_set_labels.csv")
test_set_features = pd.read_csv("test_set_features.csv")


# In[8]:


training_set_features.set_index("respondent_id", inplace = True)
training_set_labels.set_index("respondent_id", inplace = True)
test_set_features.set_index("respondent_id", inplace = True)


# In[12]:


training_set_features.head()


# In[10]:


training_set_labels.head()


# In[11]:


test_set_features.head()


# In[17]:


numeric_cols = ["xyz_concern", "xyz_knowledge", "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask", "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home", "behavioral_touch_face", "doctor_recc_xyz", "doctor_recc_seasonal", "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance", "opinion_xyz_vacc_effective", "opinion_xyz_risk", "opinion_xyz_sick_from_vacc", "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",  "household_adults", "household_children"]
non_numeric_cols = ["age_group", "education", "race", "sex", "income_poverty", "marital_status", "rent_or_own", "employment_status", "hhs_geo_region", "census_msa", "employment_industry", "employment_occupation"]
numeric = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
non_numeric = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric, numeric_cols), ('non_num', non_numeric, non_numeric_cols)], remainder = "passthrough")


# In[18]:


X_train = training_set_features
y_train = training_set_labels
X_test = test_set_features
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.fit_transform(X_test)


# In[159]:


X_train_split, X_split, y_train_split, y_split = train_test_split(X_train_preprocessed, y_train, test_size=0.09, random_state=39)
rf_model = RandomForestClassifier(n_estimators=300, random_state=39)
multi_target_model = MultiOutputClassifier(rf_model, n_jobs=-1)
multi_target_model.fit(X_train_split, y_train_split)
y_prediction = multi_target_model.predict_proba(X_split)
y_prediction_array = np.array(y_prediction)[:,:,1].T


# xyz = roc_auc_score(y_split['xyz_vaccine'], y_prediction_array[:,0])
# print('ROC AUC FOR XYZ VACCINE IS:', xyz)
# seasonal = roc_auc_score(y_split['seasonal_vaccine'], y_prediction_array[:,1])
# print('ROC AUC FOR SEASONAL VACCINE IS:', seasonal)

# In[183]:


rf_model = RandomForestClassifier(n_estimators=300, random_state=39, n_jobs=2)
multi_target_model = MultiOutputClassifier(rf_model, n_jobs=2)
para_grid = {'estimator__n_estimators': [200, 300], 'estimator__max_depth': [None, 10], 'estimator__min_samples_split': [5, 7], 'estimator__min_samples_leaf': [1, 2]}
grid_search = GridSearchCV(multi_target_model, para_grid, cv=7, scoring='roc_auc', n_jobs=2)
grid_search.fit(X_train_split, y_train_split)
best_para = grid_search.best_params_
best_score = grid_search.best_score_
print('Best parameters are: ', best_para)
print('Best ROC AUC score from GridSearchCV is: ', best_score)


# In[184]:


best_model = grid_search.best_estimator_
y_prediction = best_model.predict_proba(X_test_preprocessed)
y_prediction_array = np.array(y_prediction)[:,:,1].T


# In[185]:


final = pd.DataFrame({'respondent_id': test_set_features.index, 'xyz_vaccine': y_prediction_array[:,0], 'seasonal_vaccine': y_prediction_array[:,1]})
final.to_csv('final.csv', index=False)
final.head()


# In[ ]:




