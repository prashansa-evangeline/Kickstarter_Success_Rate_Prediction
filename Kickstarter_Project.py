#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
     


# # Data Pre-processing

# In[2]:


import pandas as pd
df = pd.read_csv(r'C:\Users\Checkout\Downloads\kickstarter_data_with_features.csv')

df.head()
     


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.nunique()


# In[7]:


df.isnull().sum()


# In[8]:


df.fillna(0, inplace=True)


# In[9]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# # EDA

# This method returns a square matrix where each cell represents the correlation between two variables in the DataFrame. The correlation coefficient can range from -1 to 1, where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.

# In[12]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# Calculate the correlation matrix
corr_matrix = df.corr()

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(50,50))

# Create the heatmap using the 'viridis' colormap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# Show the plot
plt.show()


# # state

# In[13]:


df["state"].value_counts()


# In[14]:


percentile_success = round(df["state"].value_counts() / len(df["state"]) * 100,2)

print("State Percentile in %: ")
print(percentile_success)

plt.figure(figsize = (20,5))

ax1 = sns.countplot(x="state", data=df)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
ax1.set_title("Status Project Distribuition", fontsize=15)
ax1.set_xlabel("State Description", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)

plt.show()


# # Country

# In[15]:


percentile_success = round(df["country"].value_counts() / len(df["country"]) * 100,2)

print("country Percentile in %: ")
print(percentile_success)

top = df["country"].value_counts()
plt.figure(figsize=(10,10))
gv = sns.barplot(y = top.index, x = top.values,data = df, palette = "CMRmap")
plt.title("Status Project Distribuition",fontsize = 19)
plt.show()


# # Category

# In[16]:


df["category"].value_counts()


# In[17]:


percentile_success = round(df["category"].value_counts() / len(df["category"]) * 100,2)

print("category Percentile in %: ")
print(percentile_success)

top = df["category"].value_counts()
plt.figure(figsize=(10,50))
gv = sns.barplot(y = top.index, x = top.values,data = df, palette = "CMRmap")
plt.title("Status Project Distribuition",fontsize = 19)
plt.show()
     


# # Currency

# In[18]:


df["currency"].value_counts()


# In[19]:


percentile_success = round(df["currency"].value_counts() / len(df["currency"]) * 100,2)

print("currency Percentile in %: ")
print(percentile_success)

top = df["currency"].value_counts()
plt.figure(figsize=(10,5))
gv = sns.barplot(y = top.index, x = top.values,data = df, palette = "CMRmap")
plt.title("Status Project Distribuition",fontsize = 19)
plt.show()
     


# In[20]:


# Normalization to understand the distribuition of the pledge

df["pledge_log"] = np.log(df["pledged"] + 1)
df["goal_log"] = np.log(df["goal"]+ 1)

df_failed = df[df["state"] == "failed"]
df_success = df[df["state"] == "successful"]
df_suspended = df[df["state"] == "suspended"]

plt.figure(figsize = (14,6))
plt.subplot(221)
g = sns.distplot(df["pledge_log"])
g.set_title("Pledged Log", fontsize=18)

plt.subplot(222)
g1 = sns.distplot(df["goal_log"])
g1.set_title("goal Log", fontsize=18)

plt.subplot(212)
g2 = sns.distplot(df_failed['goal_log'], color='r')
g2 = sns.distplot(df_success['goal_log'], color='b')
g2.set_title("Pledged x Goal cross distribuition", fontsize=18)

plt.subplots_adjust(wspace = 0.1, hspace = 0.4,top = 0.9)

plt.show()
     


# In[ ]:





# # Description of Goal and Pledged values

# In[21]:



print("Min Goal and Pledged values")
print(df[["goal", "pledged"]].min())
print("")
print("Mean Goal and Pledged values")
print(round(df[["goal", "pledged"]].mean(),2))
print("")
print("Median Goal and Pledged values")
print(df[["goal", "pledged"]].median())
print("")
print("Max Goal and Pledged values")
print(df[["goal", "pledged"]].max())
print("")
print("Std Goal and Pledged values")
print(round(df[["goal", "pledged"]].std(),2))


# # Understanding of Goal and pleged by its State
# 

# In[22]:


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.75, top = 0.75)

ax1 = plt.subplot(221)
ax1 = sns.boxplot(x="state", y="pledge_log", data=df, palette="hls")
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
ax1.set_title("Understanding the Pledged values by state", fontsize=15)
ax1.set_xlabel("State Description", fontsize=12)
ax1.set_ylabel("Pledged Values(log)", fontsize=12)

ax2 = plt.subplot(222)
ax2 = sns.boxplot(x="state", y="goal_log", data=df)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45)
ax2.set_title("Understanding the Goal values by state", fontsize=15)
ax2.set_xlabel("State Description", fontsize=12)
ax2.set_ylabel("Goal Values(log)", fontsize=12)

ax0 = plt.subplot(212)
ax0 = sns.regplot(x="goal_log", y="pledge_log", data=df, x_jitter=False)
ax0.set_title("Better view of Goal x Pledged values", fontsize=15)
ax0.set_xlabel("Goal Values(log)")
ax0.set_ylabel("Pledged Values(log)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)
plt.show()


# In[ ]:





# In[ ]:





# # How many category Fail and Succeed?
# 

# In[23]:


main_cats = df["category"].value_counts()
main_cats_failed = df[df["state"] == "failed"]["category"].value_counts()
main_cats_sucess = df[df["state"] == "successful"]["category"].value_counts()

plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.9, top = 0.75)

ax0 = plt.subplot(221)
ax0 = sns.barplot(x=main_cats_failed.index, y= main_cats_failed.values, orient='v')
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)
ax0.set_title("Frequency Failed by  Category", fontsize=15)
ax0.set_xlabel(" Category Failed", fontsize=12)
ax0.set_ylabel("Count", fontsize=12)

ax1 = plt.subplot(222)
ax1 = sns.barplot(x=main_cats_sucess.index, y = main_cats_sucess.values, orient='v')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("Frequency Successful by  Category", fontsize=15)
ax1.set_xlabel(" Category Sucessful", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)

ax2 = plt.subplot(212)
ax2 = sns.boxplot(x="category", y="goal_log", data=df)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(ax1.get_xticklabels(), rotation=90)
#ax2.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax2.set_title("Distribuition goal(log) by General Category", fontsize=15)
ax2.set_xlabel("Category", fontsize=12)
ax2.set_ylabel("Goal(log)", fontsize=8)
plt.show()


# # Look Goal and Pledged Means by State
# 

# In[24]:


print("Looking Goal and Pledged Mean by state ")
print(round(df.groupby(["state"])["goal", "pledged"].mean(),2))


# # Distribuition in category values as a success or failure
# 

# In[ ]:





# In[ ]:





# In[25]:


categorys_failed = df[df["state"] == "failed"]["category"].value_counts()[:25]
categorys_sucessful = df[df["state"] == "successful"]["category"].value_counts()[:25]

fig, ax = plt.subplots(ncols=2, figsize=(15,20))
plt.subplots_adjust(wspace = 0.35, top = 0.5)

g1 = plt.subplot(222)
g1 = sns.barplot(x= categorys_failed.values, y=categorys_failed.index, orient='h')
g1.set_title("Failed Category's", fontsize=15)
g1.set_xlabel("Count ", fontsize=12)
g1.set_ylabel("Category's Failed", fontsize=12)

g2 = plt.subplot(221)
g2 = sns.barplot(x= categorys_sucessful.values, y=categorys_sucessful.index, orient='h')
g2.set_title("Sucessful Category's", fontsize=15)
g2.set_xlabel("Count ", fontsize=12)
g2.set_ylabel("Category's Successful", fontsize=12)

ax2 = plt.subplot(212)
ax2 = sns.countplot(x="category", data=df)
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("General Category's", fontsize=15)
ax2.set_xlabel("Category", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)

plt.show()


# # Date & Time

# In[26]:



df['launched_at'] = pd.to_datetime(df['launched_at'])
df['laun_month_year'] = df['launched_at'].dt.to_period("M")
df['laun_year'] = df['launched_at'].dt.to_period("A")

df['deadline'] = pd.to_datetime(df['deadline'])
df['dead_month_year'] = df['deadline'].dt.to_period("M")
df['dead_year'] = df['deadline'].dt.to_period("A")
     


# In[27]:


df.laun_month_year = df.laun_month_year.dt.strftime('%Y-%m')
df.laun_year = df.laun_year.dt.strftime('%Y')
     


# In[ ]:





# In[28]:


year = df['laun_year'].value_counts()
month = df['laun_month_year'].value_counts()

fig, ax = plt.subplots(2,1, figsize=(12,10))

ax1 = sns.boxplot(x="laun_year", y='pledge_log', data=df, ax=ax[0])
ax1.set_title("Project Pledged by Year", fontsize=15)
ax1.set_xlabel("Years", fontsize=12)
ax1.set_ylabel("Pledged(log)", fontsize=12)

ax2 = sns.countplot(x="laun_year", hue='state', data=df, ax=ax[1])
ax2.set_title("Projects count by Year", fontsize=18)
ax2.set_xlabel("State columns by Year", fontsize=15)
ax2.set_ylabel("Count", fontsize=15)

plt.show()

print("Descriptive status count by year")
print(pd.crosstab(df.laun_year, df.state))
     


# In[ ]:





# # Creating a new feature to calculate percentage of pledged / goal
# 

# In[29]:


df['diff_pleded_goal'] = round(df['pledge_log'] / df['goal_log'] * 100,2)
df['diff_pleded_goal'] = df['diff_pleded_goal'].astype(float)
     


# In[30]:


plt.figure(figsize = (12,6))
sns.distplot(df[(df['diff_pleded_goal'] < 200) & (df['state'] == 'failed')]['diff_pleded_goal'], color='r')
sns.distplot(df[(df['diff_pleded_goal'] < 200) & (df['state'] == 'successful')]['diff_pleded_goal'],color='g')
plt.show()


# In[31]:


plt.figure(figsize = (18,15))

plt.subplots_adjust(hspace = 0.35, top = 0.8)

g1 = plt.subplot(211)
g1 = sns.countplot(x="laun_month_year", data=df[df['laun_month_year'] >= '2010-01'])
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Value Distribuition by Date Distribuition", fontsize=30)
g1.set_xlabel("Date Distribuition", fontsize=20)
g1.set_ylabel("Count", fontsize=20)

g2 = plt.subplot(212)
g2 = sns.boxplot(x="laun_year", y="diff_pleded_goal",data=df[df['diff_pleded_goal'] < 150], hue="state")
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Value Distribuition by Date Distribuition", fontsize=20)
g2.set_xlabel("Date Distribuition", fontsize=20)
g2.set_ylabel("Goal x Pledged (%)", fontsize=20)
plt.show()


# In[32]:


df['backers_log'] = np.log(df['backers_count'] + 1 ) 
#The + 1 is to normalize the zero or negative values

plt.figure(figsize = (8,6))
sns.distplot(df['backers_log'])

plt.show()
     


# In[33]:


plt.figure(figsize = (12,8))

plt.subplot(211)
g = sns.boxplot(x='state',y='backers_log', data=df)
g.set_title("Backers by STATE", fontsize=18)

plt.subplot(212)
g = sns.boxplot(x='category',y='backers_log', data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.show()


# In[34]:


plt.figure(figsize = (12,8))

plt.subplot(211)
g = sns.boxplot(x='laun_year',y='backers_log',data=df)
g.set_title("Backers by STATE", fontsize=18)

plt.show()
     


# In[35]:


#Looking the relation of Backers and % of goal reached
sns.lmplot(x='diff_pleded_goal', y ='backers_log', data=df[df['diff_pleded_goal'] < 150], aspect = 2,hue='state')
plt.show()
     


# In[36]:


df.head()


# In[37]:


df.info()


# In[38]:


df['Dead_Dates'] = pd.to_datetime(df['deadline']).dt.date
df['Dead_Time'] = pd.to_datetime(df['deadline']).dt.time
df['dead_hour'] = pd.to_datetime(df['deadline']).dt.hour
df['dead_minute'] = pd.to_datetime(df['deadline']).dt.minute
df['dead_day'] = pd.to_datetime(df['deadline']).dt.day
df['dead_month'] = pd.to_datetime(df['deadline']).dt.month
df['dead_year'] = pd.to_datetime(df['deadline']).dt.year
     


# In[39]:


df['launch_Dates'] = pd.to_datetime(df['launched_at']).dt.date
df['launch_Time'] = pd.to_datetime(df['launched_at']).dt.time
df['launch_hour'] = pd.to_datetime(df['launched_at']).dt.hour
df['launch_minute'] = pd.to_datetime(df['launched_at']).dt.minute
df['launch_day'] = pd.to_datetime(df['launched_at']).dt.day
df['launch_month'] = pd.to_datetime(df['launched_at']).dt.month
df['launch_year'] = pd.to_datetime(df['launched_at']).dt.year


# In[40]:


df['deadline'] = df['deadline'].apply(pd.to_datetime)
df['launched'] = df['launched_at'].apply(pd.to_datetime)

df['duration'] = df['deadline'] - df['launched_at']
df['duration'] = df['duration'].dt.days
     


# In[41]:


df['duration'] = df['duration']
     


# In[42]:


df.drop(columns=["id","photo","blurb","slug","disable_communication","currency_symbol","creator","location","profile","spotlight","urls","source_url","friends","is_starred","is_backing","permissions","deadline","launched","pledge_log","goal_log","laun_month_year","laun_year","dead_month_year","diff_pleded_goal","backers_log","Dead_Dates","Dead_Time","launch_Dates","launch_Time"],inplace=True)


# In[43]:


df.head()


# In[44]:


df=df.drop(['currency_trailing_code'],axis=1)


# In[45]:


df=df.drop(['staff_pick'],axis=1)


# In[46]:


df=df.drop(['state_changed_at','created_at','launched_at'],axis=1)


# In[47]:


df=df.drop(['deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday'],axis=1)


# In[48]:


df=df.drop(['create_to_launch','launch_to_deadline','launch_to_state_change'],axis=1)


# In[49]:


df.head()


# In[50]:


df.info()


# In[51]:


df.describe()


# In[52]:


df.info()


# # Text Preprocessing
# 

# In[53]:


get_ipython().system('pip install matplotlib-venn')


# In[54]:


get_ipython().system('pip install stop-words')


# In[55]:


pip install wordcloud


# In[56]:


import nltk
nltk.download('stopwords')
import nltk.tokenize as word_tokenize
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from stop_words import get_stop_words


# In[57]:


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=500,
    max_font_size=200, 
    width=1000, height=800,
    random_state=42,
).generate(" ".join(df['name'].dropna().astype(str)))

print(wordcloud)
plt.figure(figsize = (6, 6), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[58]:


import nltk
nltk.download('punkt')
     


# In[59]:


# convert sentance to word contain special characters.
def sentance_to_word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words

# convert whole eassy to word list
def essay_to_word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sentance_to_word(i))
    return final_words
        
# Calculate number of words in essay
def number_Of_Words(essay):
    count=0
    for i in essay_to_word(essay):
        count+=len(i)
    return count

# calculate number of character in essay
def number_Of_Char(essay):
    count=0
    for i in essay_to_word(essay):
        for j in i:
            count+=len(j)
    return count

# calculate average of words in essay
def avg_word_len(essay):
    return number_Of_Char(essay)/number_Of_Words(essay)


# In[60]:


# update dataset by calculating char_count, word_count, avg_word_len
df = df.copy()
df['name_char_count'] = df['name'].apply(number_Of_Char)
df['name_word_count'] = df['name'].apply(number_Of_Words)
df['name_avg_word_len'] = df['name'].apply(avg_word_len)
df.head()


# # Encoding
# 

# In[61]:


df= df.loc[df['state'].isin(['failed','successful','canceled','suspended','live'])]


# In[62]:


target = {'failed': 0, 'successful': 1 , 'canceled': 0, 'live':0,'suspended':0 }
df['state'] = df['state'].map(target)
     


# In[63]:


df['state'].value_counts()


# In[64]:


#Mapping countries and replacing 'N,0"' according to currency
country = {'USD':'US', 'AUD':'AU', 'CAD':'CA', 'GBP':'GB', 'EUR':'DE', 'SEK':'SE', 'DKK':'DK', 'NZD':'NZ', 'NOK':'NO', 'CHF':'CH'}
invalid = df[df['country'] == 'N,0"']
invalid['country'] = invalid['currency'].map(country)

#Placing it in original data
invalid_country = invalid['country'].iloc[:].values
j=0
for i in invalid.index:
    df['country'].iloc[i] =invalid_country[j]
    j=j+1


# In[65]:


df['category'] = df['category'].astype(str)
df['currency'] = df['currency'].astype(str)
df['country'] = df['country'].astype(str)


# In[66]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
df['category'] = labelencoder.fit_transform(df['category'])
df['currency'] = labelencoder.fit_transform(df['currency'])
df['country'] = labelencoder.fit_transform(df['country'])


# In[67]:


df['category'] = df['category'].astype(float)
df['currency'] = df['currency'].astype(float)
df['country'] = df['country'].astype(float)
     


# In[68]:


df.head(1)


# In[69]:


df.drop(columns=['name'],inplace=True)


# In[70]:


df.head(1)


# In[71]:


df.info()


# # Model Selection and Evaluation

# In[72]:


df.to_csv("final_preprocessed.csv")


# In[73]:


print(df.info())


# In[74]:


print(list(df.columns))


# In[ ]:





# In[75]:


y = df['state'].values
X = df.drop(['state'],axis=1)


# In[76]:


# Remove missing values from both X and y
X = X.dropna()
y = y[X.index]


# In[77]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)
     


# In[78]:


df.head(1)


# In[79]:


df.info()


# # Logistic Regression

# In[80]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression(penalty="l2")
logistic_reg.fit(X_train,y_train)
     


# In[81]:


y_train_pred = logistic_reg.predict(X_train)
y_test_pred = logistic_reg.predict(X_test)

d = pd.DataFrame({'Actual state': y_test, 'Predicted state': y_test_pred})
d.head(5)
     


# In[82]:


from sklearn.metrics import roc_curve,auc

train_fpr_tfidf, train_tpr_tfidf, tr_thresholds_tfidf = roc_curve(y_train, y_train_pred)
test_fpr_tfidf, test_tpr_tfidf, te_thresholds_tfidf = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr_tfidf, train_tpr_tfidf, label="Train AUC ="+str(auc(train_fpr_tfidf, train_tpr_tfidf)))
plt.plot(test_fpr_tfidf, test_tpr_tfidf, label="Test AUC ="+str(auc(test_fpr_tfidf, test_tpr_tfidf)))

plt.legend()
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()
     


# In[83]:


def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
     


# In[84]:


print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds_tfidf, train_fpr_tfidf, train_tpr_tfidf)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))
     


# In[85]:


confusion_matrix_train = confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_train = pd.DataFrame(confusion_matrix_train, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_train.index = ['Actual NO', 'Actual YES']
confusion_matrix_train.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_train, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[86]:


confusion_matrix_test = confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_test = pd.DataFrame(confusion_matrix_test, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_test.index = ['Actual NO', 'Actual YES']
confusion_matrix_test.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_test, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[87]:


from sklearn.metrics import precision_score
precision_score_train = precision_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Precision_Score of Train: ",precision_score_train))
     


# In[88]:


from sklearn.metrics import recall_score
recall_score_train = recall_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Recall_Score of Train: ",recall_score_train))
     


# In[89]:


from sklearn.metrics import f1_score
f1_score_train  = f1_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("F1_Score of Train: ",f1_score_train)


# In[90]:


from sklearn.metrics import accuracy_score
auc_score_train_lr  = accuracy_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("Accuracy_Score of Train: ",auc_score_train_lr)
     


# In[ ]:





# In[ ]:





# In[91]:


from sklearn.metrics import accuracy_score
auc_score_test_lr  = accuracy_score(y_test, predict_with_best_t(y_test_pred, best_t))
print("Accuracy_Score of Test: ",auc_score_test_lr)


# In[ ]:





# # Decision Tree

# In[92]:


def batch_predict(clf, data):

    y_data_pred = []
    tr_loop = data.shape[0] - data.shape[0]%1000
    for i in range(0, tr_loop, 1000):
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])

    if data.shape[0]%1000 !=0:
        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
    return y_data_pred
     


# In[93]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
 
min_samples_split=[5, 10, 100, 500]
max_depth=[1, 5, 10, 50]
train_auc = []
test_auc = []
for i in min_samples_split:
  for j in max_depth:
    print("min_samples_split = ", i ,'and max_depth = ',j)
    dt = DecisionTreeClassifier(min_samples_split=i, max_depth=j,class_weight='balanced')
    dt.fit(X_train, y_train)

    y_train_pred = batch_predict(dt, X_train)
    y_test_pred = batch_predict(dt, X_test)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs        
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    test_auc.append(roc_auc_score(y_test, y_test_pred))
     
print('='*100)
print('Train AUC : ',train_auc)
print('='*100)
print('Test AUC : ',test_auc)
print('='*100)


# In[94]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

min_samples_split=[5, 10, 100, 500]
max_depth= [1, 5, 10, 50]

dt=DecisionTreeClassifier(class_weight='balanced')
params={'max_depth': [1, 5, 10], 'min_samples_split':[5, 10, 100]}

cross_val=GridSearchCV(estimator=dt, param_grid=params, cv=3, scoring='roc_auc',return_train_score=True,verbose=2,n_jobs=-1)

cross_val.fit(X_train,y_train)
train_auc= cross_val.cv_results_['mean_train_score']
train_auc_std= cross_val.cv_results_['std_train_score']
test_auc = cross_val.cv_results_['mean_test_score'] 
test_auc_std= cross_val.cv_results_['std_test_score']

print('='*100)
print('The Best Parameters are : ',cross_val.best_params_)
print('='*100)
     


# In[95]:


sns.set()
max_scores = pd.DataFrame(cross_val.cv_results_).groupby(['param_min_samples_split', 'param_max_depth']).max().unstack()[['mean_test_score', 'mean_train_score']]
fig, ax = plt.subplots(1,2, figsize=(20,6))
sns.heatmap(max_scores.mean_train_score, annot = True, fmt='.4g', ax=ax[0])
sns.heatmap(max_scores.mean_test_score, annot = True, fmt='.4g', ax=ax[1])
ax[0].set_title('Train Set')
ax[1].set_title('Test Set')
plt.show()


# In[96]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
print(cross_val.best_params_)
dt_best = cross_val.best_estimator_
dt_best.fit(X_train, y_train)

y_train_pred = batch_predict(dt_best, X_train)
y_test_pred = batch_predict(dt_best, X_test)

from sklearn.metrics import mean_squared_error as mse , r2_score
from math import sqrt
rms = sqrt(mse(y_test, y_test_pred))
score = r2_score(y_test, y_test_pred)

print(rms)
print(score)

xxx = pd.DataFrame({'Actual state': y_test, 'Predicted state': y_test_pred})
xxx.head(5)


# In[97]:


from sklearn.metrics import roc_curve, roc_auc_score, auc

train_fpr_tfidf, train_tpr_tfidf, tr_thresholds_tfidf = roc_curve(y_train, y_train_pred)
test_fpr_tfidf, test_tpr_tfidf, te_thresholds_tfidf = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr_tfidf, train_tpr_tfidf, label="Train AUC ="+str(auc(train_fpr_tfidf, train_tpr_tfidf)))
plt.plot(test_fpr_tfidf, test_tpr_tfidf, label="Test AUC ="+str(auc(test_fpr_tfidf, test_tpr_tfidf)))

plt.legend()
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()
     


# In[98]:


# we are writing our own function for predict, with defined thresould
# we will pick a threshold that will give the least fpr
def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
     


# In[99]:


print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds_tfidf, train_fpr_tfidf, train_tpr_tfidf)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))
     


# In[100]:


confusion_matrix_train = confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_train = pd.DataFrame(confusion_matrix_train, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_train.index = ['Actual NO', 'Actual YES']
confusion_matrix_train.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_train, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[101]:



confusion_matrix_test = confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_test = pd.DataFrame(confusion_matrix_test, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_test.index = ['Actual NO', 'Actual YES']
confusion_matrix_test.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_test, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[102]:


from sklearn.metrics import precision_score
precision_score_train = precision_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Precision_Score of Train: ",precision_score_train))
     


# In[103]:


from sklearn.metrics import recall_score
recall_score_train = recall_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Recall_Score of Train: ",recall_score_train))
     


# In[104]:


from sklearn.metrics import f1_score
f1_score_train  = f1_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("F1_Score of Train: ",f1_score_train)


# In[105]:


from sklearn.metrics import accuracy_score
auc_score_train_dt  = accuracy_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("Accuracy_Score of Train: ",auc_score_train_dt)
     


# In[ ]:





# In[106]:


from sklearn.metrics import accuracy_score
auc_score_test_dt  = accuracy_score(y_test, predict_with_best_t(y_test_pred, best_t))
print("Accuracy_Score of Test: ",auc_score_test_dt)


# In[ ]:





# In[ ]:





# #  Gradient Boosting Decision Tree
# 

# In[107]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

max_depth= [1, 2, 3]
learning_rate=[0.001,0.01,1]

gbdt=GradientBoostingClassifier()
params={'max_depth': [1, 2, 3], 'learning_rate':[0.001,0.01,1]}

grid=GridSearchCV(estimator=gbdt, param_grid=params, cv=3, scoring='roc_auc',return_train_score=True,verbose=2,n_jobs=-1)

grid.fit(X_train,y_train)
train_auc= grid.cv_results_['mean_train_score']
train_auc_std= grid.cv_results_['std_train_score']
test_auc = grid.cv_results_['mean_test_score'] 
test_auc_std= grid.cv_results_['std_test_score']

print('='*100)
print('The Best Parameters are : ',grid.best_params_)
print('='*100)


# In[108]:


sns.set()
max_scores = pd.DataFrame(grid.cv_results_).groupby(['param_max_depth', 'param_learning_rate']).max().unstack()[['mean_test_score', 'mean_train_score']]
fig, ax = plt.subplots(1,2, figsize=(20,6))
sns.heatmap(max_scores.mean_train_score, annot = True, fmt='.4g', ax=ax[0])
sns.heatmap(max_scores.mean_test_score, annot = True, fmt='.4g', ax=ax[1])
ax[0].set_title('Train Set')
ax[1].set_title('Test Set')
plt.show()
     


# In[109]:


gbdt_best = grid.best_estimator_
gbdt_best.fit(X_train, y_train)

y_train_pred = batch_predict(gbdt_best, X_train)
y_test_pred = batch_predict(gbdt_best, X_test)

from sklearn.metrics import mean_squared_error as mse , r2_score
from math import sqrt
rms = sqrt(mse(y_test, y_test_pred))
score = r2_score(y_test, y_test_pred)

print(rms)
print(score)

xxx = pd.DataFrame({'Actual state': y_test, 'Predicted state': y_test_pred})
xxx.head(5)
     


# In[110]:


train_fpr_tfidf, train_tpr_tfidf, tr_thresholds_tfidf = roc_curve(y_train, y_train_pred)
test_fpr_tfidf, test_tpr_tfidf, te_thresholds_tfidf = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr_tfidf, train_tpr_tfidf, label="Train AUC ="+str(auc(train_fpr_tfidf, train_tpr_tfidf)))
plt.plot(test_fpr_tfidf, test_tpr_tfidf, label="Test AUC ="+str(auc(test_fpr_tfidf, test_tpr_tfidf)))

plt.legend()
plt.xlabel("False Positive Rate(FPR)")
plt.ylabel("True Positive Rate(TPR)")
plt.title("ERROR PLOTS")
plt.grid(True)
plt.show()


# In[111]:


print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds_tfidf, train_fpr_tfidf, train_tpr_tfidf)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))


# In[112]:


confusion_matrix_train = confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_train = pd.DataFrame(confusion_matrix_train, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_train.index = ['Actual NO', 'Actual YES']
confusion_matrix_train.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_train, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[113]:



confusion_matrix_test = confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))
y_true = ['NO','YES']
y_pred = ['NO','YES']
confusion_matrix_test = pd.DataFrame(confusion_matrix_test, columns=np.unique(y_true), index = np.unique(y_true))
confusion_matrix_test.index = ['Actual NO', 'Actual YES']
confusion_matrix_test.columns = ['Predicted NO','Predicted YES']
sns.heatmap(confusion_matrix_test, annot=True,annot_kws={"size": 20},fmt="d",cmap='Blues')


# In[114]:


from sklearn.metrics import precision_score
precision_score_train = precision_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Precision_Score of Train: ",precision_score_train))
     


# In[115]:


from sklearn.metrics import recall_score
recall_score_train = recall_score(y_train, predict_with_best_t(y_train_pred, best_t))
print(print("Recall_Score of Train: ",recall_score_train))
     


# In[116]:


from sklearn.metrics import f1_score
f1_score_train  = f1_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("F1_Score of Train: ",f1_score_train)


# In[117]:


from sklearn.metrics import accuracy_score
auc_score_train_gbdt  = accuracy_score(y_train, predict_with_best_t(y_train_pred, best_t))
print("Accuracy_Score of Train: ",auc_score_train_gbdt)
     


# In[ ]:





# In[118]:


from sklearn.metrics import accuracy_score
auc_score_test_gbdt  = accuracy_score(y_test, predict_with_best_t(y_test_pred, best_t))
print("Accuracy_Score of Test: ",auc_score_test_gbdt)


# In[ ]:





# In[119]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
my_data = pd.DataFrame({
    'cat': ['Logistic regression', 'Decision Tree ', 'Gradient boosting classifier'], 
    'val': [auc_score_train_lr, auc_score_train_dt, auc_score_train_gbdt],
    'color': ['#FFC0CB', '#FFC0CB', '#FFA500']
})

# Create the barplot
ax = sns.barplot(x='val', y='cat', data=my_data, palette=my_data['color'])

# Set the axis labels
ax.set(xlabel='Performance of categorical models', ylabel='Models')

# Show the plot
plt.show()


# In[ ]:




