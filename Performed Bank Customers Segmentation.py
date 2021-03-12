#!/usr/bin/env python
# coding: utf-8

# # Understanding the Problem Statement and Business Case
# 
# - Marketing is crucial for the growth and sustainability of any business.
# - Marketers can help build the company's brand, engage customers, grow revenue, and increase sales.
# - Marketers empower business growth by reaching new customers
# - Marketers educate and communicate value proposition to customers
# - Marketers drive sales and traffic to products/services
# - Marketers engage customers and understand their needs
# 
# - One of the key pain points for marketers is to know their customers and identify their needs.
# - By understanding the customer, marketers can launch a targeted marketing campaign that is tailored for specific needs.
# - If data about the customers is available, data science can be applied to perform market segmentation.
# 
# - In this case study, I am a consultant to a bank in New York City.
# - The bank has extensive data on their customers for the past 6 months.
# - The marketing team at the bank wants to launch a targeted ad marketing campaign by dividing their customers into at least 3 distinctive groups.
# 
# # Data Description:
# 
# - CUSTID: Identification of Credit Card holder 
# - BALANCE: Balance amount left in customer's account to make purchases
# - BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# - PURCHASES: Amount of purchases made from account
# - ONEOFFPURCHASES: Maximum purchase amount done in one-go
# - INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# - CASH_ADVANCE: Cash in advance given by the user
# - PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# - ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# - PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# - CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
# - CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
# - PURCHASES_TRX: Number of purchase transactions made
# - CREDIT_LIMIT: Limit of Credit Card for user
# - PAYMENTS: Amount of Payment done by user
# - MINIMUM_PAYMENTS: Minimum amount of payments made by user  
# - PRC_FULL_PAYMENT: Percent of full payment paid by user
# - TENURE: Tenure of credit card service for user
# 
# # Import Libraries and Datasets

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from keras.optimizers import SGD


# In[2]:


# You have to include the full link to the csv file containing your dataset
creditcard_df = pd.read_csv("D:\Python and Machine Learning for Financial Analysis\Marketing_data.csv")


# In[3]:


creditcard_df


# In[4]:


creditcard_df.info()


# - 18 features with 8950 points  

# In[5]:


creditcard_df.describe()


# - Mean balance is \\$1564
# - Balance frequency is frequently updated on average ~0.9
# - Purchases average is \\$1000
# - One off purchase average is ~\\$600
# - Average purchases frequency is around 0.5
# - Average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
# - Average credit limit ~ 4500
# - Percent of full payment is 15%
# - Average tenure is 11 years

# In[6]:


# Let's see who made one off purchase of $40761!
creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == 40761.25]


# In[7]:


creditcard_df['CASH_ADVANCE'].max()


# In[8]:


# Let's see who made cash advance of $47137!
creditcard_df[creditcard_df['CASH_ADVANCE'] == 47137.211760000006]


# - This customer made 123 cash advance transactions!!
# - Never paid credit card in full
# 
# # Visualized and Explored Dataset

# In[9]:


# Let's see if we have any missing data, luckily we don't!
sns.heatmap(creditcard_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[10]:


creditcard_df.isnull().sum()


# In[11]:


# Filled up the missing elements with mean of the 'MINIMUM_PAYMENT' 
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()


# In[12]:


# Filled up the missing elements with mean of the 'CREDIT_LIMIT' 
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()


# In[13]:


sns.heatmap(creditcard_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[14]:


# Let's see if we have duplicated entries in the data
creditcard_df.duplicated().sum()


# In[15]:


# Let's drop Customer ID since it has no meaning here 
creditcard_df.drop("CUST_ID", axis = 1, inplace= True)


# In[16]:


creditcard_df.head()


# In[17]:


n = len(creditcard_df.columns)
n


# In[18]:


creditcard_df.columns


# In[19]:


creditcard_df.dtypes


# In[20]:


# to change use .astype() 
#creditcard_df['TENURE'] = creditcard_df.TENURE.astype(float)


# In[21]:


# distplot combines the matplotlib.hist function with seaborn kdeplot()
# KDE Plot represents the Kernel Density Estimate
# KDE is used for visualizing the Probability Density of a continuous variable. 
# KDE demonstrates the probability density at different values in a continuous variable. 

#plt.figure(figsize = (10, 50))
#for i in range(len(creditcard_df.columns)):
 #plt.subplot(17, 1, i+1)
 #sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws = {"color" : "b", "lw" : 3, "label" : "KDE"}, hist_kws = {"color" : "g"})
 #plt.title(creditcard_df.columns[i])

#plt.tight_layout()

#sns.distplot(creditcard_df['TENURE'], kde_kws = {"color" : "b", "lw" : 3, "label" : "KDE"}, hist_kws = {"color" : "g"})


# - Mean of balance is \\$1500
# - 'Balance_Frequency' for most customers is updated frequently ~1
# - For 'PURCHASES_FREQUENCY', there are two distinct group of customers
# - For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently 
# - Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0
# - Credit limit average is around \\$4500
# - Most customers are ~11 years tenure

# In[22]:


sns.pairplot(creditcard_df)


# - Correlation between 'PURCHASES' and ONEOFF_PURCHASES & INSTALMENT_PURCHASES 
# - Trend between 'PURCHASES' and 'CREDIT_LIMIT' & 'PAYMENTS'

# In[23]:


correlations = creditcard_df.corr()


# In[24]:


f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)


# - 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments. 
# - Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'
# 
# # Understanding the Theory and Intuition behind K-Means
# 
# ## K-Means Intuition
# 
# - K-means is an unsupervised learning algorithm (clustering)
# - K-means works by grouping some data points together (clustering) in an unsupervised fashion.
# - The algorithm groups observations with similar attriubte values together by measuring the Euclidian distance between points.
# 
# ## K-Means Algorithm Steps
# 
# 1. Choose number of clusters "K"
# 2. Select random K points that are going to be the centroids for each cluster
# 3. Assign each data point to the nearest centroid, doing so will enable us to create "K" number of clusters
# 4. Calculate a new centroid for each cluster
# 5. Reassign each data point to the new closest centroid
# 6. Go to step - 4 and repeat.
# 
# # Found the Optimal Number of Clusters using Elbow Method
# 
# - The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset.
# - If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.

# In[25]:


# Let's scale the data first
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)


# In[26]:


creditcard_df_scaled.shape


# In[27]:


creditcard_df_scaled


# In[28]:


scores_1 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters = i)
  kmeans.fit(creditcard_df_scaled)
  scores_1.append(kmeans.inertia_) 

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores') 
plt.show()


# - From this we can observe that, 4th cluster seems to be forming the elbow of the curve. 
# - However, the values does not reduce linearly until 8th cluster. 
# - Let's choose the number of clusters to be 7.
# 
# # Applied K-Means Method

# In[29]:


kmeans = KMeans(8)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_


# In[30]:


kmeans.cluster_centers_.shape


# In[31]:


cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])
cluster_centers           


# In[32]:


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
cluster_centers


# - First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance (\\$104) and cash advance (\\$303), Percentage of full payment = 23%
# - Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance (\\$5000) and cash advance (~\\$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# - Third customer cluster (VIP/Prime): high credit limit \\$16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# - Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance 

# In[33]:


labels.shape # Labels associated to each data point


# In[34]:


labels.max()


# In[35]:


labels.min()


# In[36]:


y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
y_kmeans


# In[37]:


# Concatenated the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()


# In[38]:


# Plot the histogram of various clusters
for i in creditcard_df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()


# # Applied Principal Component Analysis and Visualize the Results
# 
# ## Principal Component Analysis: Overview
# 
# - PCA is an unsupervised machine learning algorithm.
# - PCA performs dimensionality reductions while attempting at keeping the original information unchanged.
# - PCA works by trying to find a new set of features called components.
# - Components are composites of the uncorrelated given input features.

# In[39]:


# Obtained the principal components 
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)
principal_comp


# In[40]:


# Created a dataframe with the two components
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()


# In[41]:


# Concatenated the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[42]:


plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
plt.show()


# # Understanding the Theory and Intuition behind Autoencoders
# 
# ## Autoencoders Intuition
# 
# - Autoencoders are a type of Artificial Neural Networks that are used to perform a task of data encoding (representation learning).
# - Autoencoders use the same input data for the input and output.
# 
# ## The Code Layer
# 
# - Autoencoders work by adding a bottleneck in the network.
# - This bottleneck forces the network to create a compressed (encoded) version of the original input
# - Autoencoders work well if correlations exists between input data (performs poorly if all the input data is independent)
# 
# # Applied Autoencoders (Performed Dimensionality Reduction using Autoencoders)

# In[43]:


encoding_dim = 7

input_df = Input(shape=(17,))

# Glorot normal initializer (Xavier normal initializer) draws samples from a truncated normal distribution 

x = Dense(encoding_dim, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer = 'glorot_uniform')(x)

x = Dense(2000, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer = 'glorot_uniform')(x)

decoded = Dense(17, kernel_initializer = 'glorot_uniform')(x)

# Autoencoder
autoencoder = Model(input_df, decoded)

#Encoder - used for our dimension reduction
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer= 'adam', loss='mean_squared_error')


# In[44]:


creditcard_df_scaled.shape


# In[45]:


autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, batch_size = 128, epochs = 25,  verbose = 1)


# In[46]:


autoencoder.save_weights('autoencoder.h5')


# In[47]:


pred = encoder.predict(creditcard_df_scaled)


# In[48]:


pred.shape


# In[49]:


scores_2 = []

range_values = range(1, 20)

for i in range_values:
  kmeans = KMeans(n_clusters= i)
  kmeans.fit(pred)
  scores_2.append(kmeans.inertia_)

plt.plot(scores_2, 'bx-')
plt.title('Finding right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('scores') 
plt.show()


# In[50]:


plt.plot(scores_1, 'bx-', color = 'r')
plt.plot(scores_2, 'bx-', color = 'g')


# In[51]:


kmeans = KMeans(4)
kmeans.fit(pred)
labels = kmeans.labels_
y_kmeans = kmeans.fit_predict(creditcard_df_scaled)


# In[52]:


df_cluster_dr = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
df_cluster_dr.head()


# In[53]:


pca = PCA(n_components=2)
prin_comp = pca.fit_transform(pred)
pca_df = pd.DataFrame(data = prin_comp, columns =['pca1','pca2'])
pca_df.head()


# In[54]:


pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[55]:


plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','yellow'])
plt.show()

