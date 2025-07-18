from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
obesity_data = fetch_ucirepo(id=544) 
  
# data (as pandas dataframes) 
X = obesity_data.data.features 
y = obesity_data.data.targets 
  
# # metadata 
# print(obesity_data.metadata) 
  
# # variable information 
# print(obesity_data.variables) 

df = pd.concat([X, y], axis=1)
print(df.head())

