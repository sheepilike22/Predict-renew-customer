# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:26:27 2020

@author: angela
"""
import  pymysql 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
################# crm cloud #############
## connect to crm cloud 
conn = pymysql.connect(
    host="***********",
    port=int(3306),
    user="*********",
    passwd="*********",
    db="*********")

##### find the target from the log ###
## time  2016.4 ~ 2019.10 (utc-create-time<20191101)
buy="SELECT i.UTC_CREATE_DATE, i.ORIGINAL_VALUE, i.NEW_VALUE, a.ENTITY_ID, s.UC_SITE_NAME , DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), DATE(i.UTC_CREATE_DATE)) AS D1,DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), STR_TO_DATE(i.ORIGINAL_VALUE,'%Y/%m/%d')) AS length FROM `com_audit_log_item` i JOIN `com_audit_log` a ON i.AUDIT_LOG_ID=a.ID  JOIN `acm_store` s ON s.ID=a.ENTITY_ID  JOIN ACM_CONTACT_MECH c ON s.PARTY_ID=c.PARTY_ID and c.SUBTYPE='AuthEmail' and a.SUBTYPE='ModificationLog'  and i.DATA_NAME='TimeUse' and i.ORIGINAL_VALUE<>'' and i.NEW_VALUE <> ''  and DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), STR_TO_DATE(i.ORIGINAL_VALUE,'%Y/%m/%d'))>=360 and not exists (  select 1 from `com_audit_log_item` ii  where i.AUDIT_LOG_ID=ii.AUDIT_LOG_ID and ii.DATA_NAME='CurrentPackage' and ii.ORIGINAL_VALUE IN ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial') and ii.NEW_VALUE in ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial')) and RIGHT(c.ELECTRONIC_ADDRESS_STRING,10) <> 'gss.com.tw' AND DATE(i.UTC_CREATE_DATE)<'2019-11-01' and  STR_TO_DATE(i.ORIGINAL_VALUE,'%Y/%m/%d')>='2016,4,1'  group by s.CURRENT_PACKAGE_ID, i.UTC_CREATE_DATE, a.ENTITY_ID order by i.UTC_CREATE_DATE, a.ENTITY_ID asc;"
# buy="SELECT i.UTC_CREATE_DATE, i.ORIGINAL_VALUE, i.NEW_VALUE, a.ENTITY_ID, s.UC_SITE_NAME , DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), DATE(i.UTC_CREATE_DATE)) AS D1,DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), STR_TO_DATE(i.ORIGINAL_VALUE,'%Y/%m/%d')) AS length FROM `com_audit_log_item` i JOIN `com_audit_log` a ON i.AUDIT_LOG_ID=a.ID  JOIN `acm_store` s ON s.ID=a.ENTITY_ID  JOIN ACM_CONTACT_MECH c ON s.PARTY_ID=c.PARTY_ID and c.SUBTYPE='AuthEmail' and a.SUBTYPE='ModificationLog'  and i.DATA_NAME='TimeUse' and i.ORIGINAL_VALUE<>'' and i.NEW_VALUE <> ''  and DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), STR_TO_DATE(i.ORIGINAL_VALUE,'%Y/%m/%d'))>=360 and not exists (  select 1 from `com_audit_log_item` ii  where i.AUDIT_LOG_ID=ii.AUDIT_LOG_ID and ii.DATA_NAME='CurrentPackage' and ii.ORIGINAL_VALUE IN ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial') and ii.NEW_VALUE in ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial')) and RIGHT(c.ELECTRONIC_ADDRESS_STRING,10) <> 'gss.com.tw'  group by s.CURRENT_PACKAGE_ID, i.UTC_CREATE_DATE, a.ENTITY_ID order by length asc;"
store_buy = pd.read_sql(buy,conn)

## store_duetime
due="SELECT sl.STORE_ID,sl.DATE_SPECIFIED,s.UTC_CREATE_DATE, s.UC_SITE_NAME,s.CURRENT_PACKAGE_ID FROM `acm_store_license` sl JOIN `acm_store` s ON sl.STORE_ID = s.ID JOIN ACM_CONTACT_MECH c ON s.PARTY_ID=c.PARTY_ID and c.SUBTYPE='AuthEmail' WHERE sl.FEATURE_TYPE='TimeUse' and s.CURRENT_PACKAGE_ID not in ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial') and RIGHT(c.ELECTRONIC_ADDRESS_STRING,10) <> 'gss.com.tw';"
store_due=pd.read_sql(due,conn)

## license (only have new_value & using >=180 days)
first_buy="SELECT i.UTC_CREATE_DATE, i.ORIGINAL_VALUE, i.NEW_VALUE, a.ENTITY_ID, s.UC_SITE_NAME , DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), DATE(i.UTC_CREATE_DATE)) AS D1 FROM `com_audit_log_item` i JOIN `com_audit_log` a ON i.AUDIT_LOG_ID=a.ID  JOIN `acm_store` s ON s.ID=a.ENTITY_ID  JOIN ACM_CONTACT_MECH c ON s.PARTY_ID=c.PARTY_ID and c.SUBTYPE='AuthEmail' and a.SUBTYPE='ModificationLog'  and i.DATA_NAME='TimeUse' and i.ORIGINAL_VALUE='' and i.NEW_VALUE <> ''  and DATEDIFF(STR_TO_DATE(i.NEW_VALUE,'%Y/%m/%d'), DATE(i.UTC_CREATE_DATE))>=180 and not exists (  select 1 from `com_audit_log_item` ii  where i.AUDIT_LOG_ID=ii.AUDIT_LOG_ID and ii.DATA_NAME='CurrentPackage'  and ii.NEW_VALUE in ('Free','VitalFree','Basic1M','Ent1M','Ent2M','Social1M','TinyEnt1M','VitalEntSmTrial','VitalEntTrial','VitalSmeTrial')) and RIGHT(c.ELECTRONIC_ADDRESS_STRING,10) <> 'gss.com.tw' AND DATE(i.UTC_CREATE_DATE)<'2019-11-01'   group by s.CURRENT_PACKAGE_ID, i.UTC_CREATE_DATE, a.ENTITY_ID order by i.UTC_CREATE_DATE, a.ENTITY_ID asc;"
store_first_buy = pd.read_sql(first_buy,conn)


############################### dataframe ########################
duetime = pd.DataFrame({'ENTITY_ID':store_due.STORE_ID,
                  'DATE_SPECIFIED':store_due.DATE_SPECIFIED}) 

## merge date_specified to store_buy
store=pd.concat([store_first_buy,store_buy],axis=0)
log_license=pd.merge(store, duetime,on='ENTITY_ID', how="left", sort=True)
log_license_nodups = log_license.drop_duplicates() # delete duplicate rows
log_license_nodups =log_license_nodups.loc[~log_license_nodups['UC_SITE_NAME'].isnull()]
## CHECK THE LICENSE IN THE HISTORY
aa=log_license_nodups.sort_values(['ENTITY_ID','UTC_CREATE_DATE'])
aa=aa.reset_index()
aa=aa.drop(columns=['index'])

## add column about next_due
next_due=aa['NEW_VALUE'].drop(index=0)
t=aa['ENTITY_ID'].drop(index=0)
next_due=next_due.reset_index()
t=t.reset_index()
next_due=next_due.drop(columns=['index'])
t=t.drop(columns=['index'])
aa['next_due']=next_due
aa['t']=t

## write the next_due  
aa.loc[aa['ENTITY_ID'] == aa['t'], 'next_due'] = aa['next_due']  
aa.loc[aa['ENTITY_ID'] != aa['t'], 'next_due'] = 'NA' 

## change the type
aa['NEW_VALUE']=pd.to_datetime( aa['NEW_VALUE'],format='%Y/%m/%d')
aa['ORIGINAL_VALUE']=pd.to_datetime( aa['ORIGINAL_VALUE'],format='%Y/%m/%d')

## first_buy add original date
aa.loc[aa['ORIGINAL_VALUE'].isnull(), 'ORIGINAL_VALUE'] = aa['UTC_CREATE_DATE']   

# SELECT the range (ga_web:20161003-20200226)
aa=aa[(aa['NEW_VALUE']<='20200226')&(aa['ORIGINAL_VALUE']>='20161003')]
aa=aa.sort_values(['ENTITY_ID','UTC_CREATE_DATE']).reset_index()  # order by date & entity_id
aa=aa.drop(columns=['index'])
aa=aa.loc[aa['DATE_SPECIFIED'] >= aa['NEW_VALUE']] # ensure newest license
## ORANGIZE THE LIST
list=aa
list=list.drop(columns=['UTC_CREATE_DATE','D1','DATE_SPECIFIED','t'])
list['target']=list['next_due']
list.loc[list['target'] == 'NA', 'target'] = 0
list.loc[list['target'] != 0, 'target'] = 1

################# ga web #####################
### USE GA DATA
ga_web=pd.read_csv('C:\\Users\\angela\\CSBU\\GA\\GA_Web\\crm_ga_20161003-20200226.csv') 
ga_web=ga_web.drop(columns=['dimension2','dimension3','dimension4','country','avgSessionDuration','bounceRate','uniquePageviews'])

ga_web['pagePathLevel2'] = ga_web['pagePathLevel2'].apply(lambda x: re.sub('/', '', x))
ga_web['pagePathLevel2'] = ga_web['pagePathLevel2'].apply(lambda x: re.sub('.mvc', '', x))
ga_web['dimension1']=ga_web['dimension1'].str.upper()  # Consistent

########### explore the data #################
## observe (every month)
store_list=pd.merge(list, store_due, on='UC_SITE_NAME', how="left", sort=True) # ADD CURRENT PACKAGE
store_list=store_list.drop(columns=['STORE_ID','DATE_SPECIFIED'])
ga=pd.merge(ga_web, store_list, left_on='dimension1', right_on='UC_SITE_NAME', how="left", sort=True)
ga=ga.loc[~ga['UC_SITE_NAME'].isnull()] # remove UC_SITE_NAME is blank
ga=ga.drop(columns=['ENTITY_ID','length','next_due'])
ga['date'] = pd.to_datetime(ga['date'], format = '%Y%m%d')
ga = ga.drop_duplicates()
ga=ga.loc[(ga['date'] <= ga['NEW_VALUE'])&(ga['date'] >= ga['ORIGINAL_VALUE'])]
ga['datediff'] = (ga['date'] -ga['ORIGINAL_VALUE']).dt.days  # timedelta to integer
ga['datediff'] =ga['datediff']//30 +1  # day to month
ga['length'] = (ga['NEW_VALUE'] -ga['UTC_CREATE_DATE']).dt.days  # timedelta to integer


ga_day=ga.groupby([ga.date, ga.dimension1, ga.NEW_VALUE]).agg({'pageviews': 'sum',
                                                               'sessions': 'sum',
                                                               'sessionDuration': 'sum',
                                                               'datediff': 'max',
                                                               'length': 'max',
                                                               'CURRENT_PACKAGE_ID': 'max',
                                                               'target': 'max'})

ga_day=ga_day.reset_index()  # remove multiple index
#### wathcing every month

ga_m=ga_day.groupby([ga.date, ga.dimension1, ga.NEW_VALUE]).agg({'pageviews': np.mean,
                                                               'sessions': np.mean,
                                                               'sessionDuration': np.mean,
                                                               'datediff': 'max',
                                                               'target': 'max'})
ga_1=ga_m[ga_m['target']==1]
ga_0=ga_m[ga_m['target']==0]
ga_1=ga_1.groupby([ga_1.datediff]).agg({'pageviews': np.mean,
                                         'sessions': np.mean,
                                         'sessionDuration': np.mean})
ga_0=ga_0.groupby([ga_0.datediff]).agg({'pageviews': np.mean,
                                         'sessions': np.mean,
                                         'sessionDuration': np.mean})

## multiple line plot
# target 1
plt.plot( ga_1.pageviews,data=ga_1, marker='', markerfacecolor='blue', color='skyblue', linewidth=2)
plt.plot( ga_1.sessions,data=ga_1, marker='', color='olive', linewidth=2)
#plt.plot( ga_1.sessionDuration,data=ga_1, marker='o', color='pink', linewidth=2)
plt.legend()
# target 0
plt.plot( ga_0.pageviews,data=ga_0, marker='', markerfacecolor='blue',  color='skyblue', linewidth=2)
plt.plot( ga_0.sessions,data=ga_0, marker='', color='olive', linewidth=2)
#plt.plot( ga_0.sessionDuration,data=ga_0, marker='o', color='pink', linewidth=2)
plt.legend()

## one month action from ga
ga_1month=ga_day[ga_day.datediff==1].groupby([ga_day.dimension1, ga_day.NEW_VALUE]).agg({'pageviews': ['sum',np.mean],
                                                                                        'sessions': ['sum',np.mean],
                                                                                        'sessionDuration': ['sum',np.mean],
                                                                                        'length': 'max',
                                                                                        'CURRENT_PACKAGE_ID': 'max',
                                                                                        'target': 'max'}) 
ga_1month.columns = [f'{x}_{y}' for x,y in ga_1month.columns]   # change the cloumn name

################## model: xgboost (train) ##################
# Load the Diabetes dataset
df= ga_1month
df=df.rename(columns={"target_max": "target", "CURRENT_PACKAGE_ID_max": "CURRENT_PACKAGE_ID"})
df = pd.get_dummies(df)   #one-hot encoding

#####訓練過程準備###########
train_labels = df['target']  
df= df.drop(columns=['target'])

## put data in the model #####
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import seaborn as sns
######分類0和1比例######
ratio = (train_labels == 0).sum()/ (train_labels == 1).sum()

X_train, X_test, y_train, y_test = train_test_split(df, train_labels, test_size=0.2, stratify=train_labels, random_state=1)

#########Apply model####### 
clf = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=0.1, subsample=0.5, scale_pos_weight=ratio )
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc', early_stopping_rounds=50)

#使用模型來預測
predictions = clf.predict(X_test)

## valid acc
(clf.predict(X_test)==y_test).mean()

#模型分數
clf.score(X_train,y_train)
evalution=classification_report(y_test,predictions)

# visualize the important information
top16 = pd.DataFrame({'features': X_train.columns, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False).head(16)
sns.barplot(x=top16['importance'], y=top16['features'])

#建立confusion_matrix
matrix=confusion_matrix(y_test,predictions)

#ROC曲線
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Perform 6-fold cross validation
scores = cross_val_score(clf, df, train_labels, cv=6)

#####################################################################
################ find the detailed information #####################
## pagepath(long -> wide)   
ga_web_action=pd.pivot_table(ga_web, index=['dimension1','date'], columns=['pagePathLevel2'], values=['pageviews', 'sessions','sessionDuration'])
ga_web_action.columns = [f'{x}_{y}' for x,y in ga_web_action.columns]   # change the cloumn name
ga_web_action=ga_web_action.reset_index()  # remove multiple index

### watch the pageLevel
store_ga=pd.merge(ga_web_action, list, left_on='dimension1', right_on='UC_SITE_NAME', how="left", sort=True)
store_ga=store_ga.drop(columns=['ENTITY_ID','length','next_due'])
store_ga=store_ga.loc[~store_ga['UC_SITE_NAME'].isnull()] # remove UC_SITE_NAME is blank
store_ga['date'] = pd.to_datetime(store_ga['date'], format = '%Y%m%d')
store_ga = store_ga.drop_duplicates()
store_ga=store_ga.loc[(store_ga['date'] <= store_ga['NEW_VALUE'])&(store_ga['date'] >= store_ga['ORIGINAL_VALUE'])]

store_ga['datediff'] = (store_ga['date'] -store_ga['ORIGINAL_VALUE']).dt.days  # timedelta to integer





