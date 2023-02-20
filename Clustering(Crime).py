c# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import the data
import pandas as pd
df = pd.read_csv("D:\\Data Science\\Assignments\\Principal Component Analysis\\wine.csv")
df
df.dtypes

df.info()

# Converting data to numpy array
df1=df.values
df1
    
# Normalizing the  numerical data

from sklearn.preprocessing import scale
Wine=scale(df1)
Wine

# Applying PCA Fit Transform to dataset

from sklearn.decomposition import PCA

pca = PCA()
pca_values = pca.fit_transform(Wine)
pca_values            
    
##Percentage of varaiance'

var=pca.explained_variance_ratio_ 
sum(pca.explained_variance_ratio_)       
        
##Graph

final_df=pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df      
        
# Visualization of PCAs
import seaborn as sns
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df);        
        
sns.scatterplot(data=final_df, x='PC1', y='PC2', hue='Type');
        
#Clustering

#Create dendograms

plt.figure(figsize=(10,8))
dendrogram=df      
        
        
        
        
        
        
        
        
        
        
        
        
        
        