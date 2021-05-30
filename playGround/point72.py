import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv("Data.csv")
tn=['t'+str(i) for i in range(1,11)]
id=np.unique(df['ID'].values)
for i in id:assert np.sum(df['ID']==i)==10,"bad data"
df['group']=-1
for t in tn:
    ids=df[df['Time']==t].ID
    #print(ids)
    assert len(np.unique(ids))==len(ids),"duplicate"
    vals=df[df['Time']==t].Value
    ans=pd.qcut(vals,4, labels=False)
    print(ans.values)
    #df[df['Time'] == t]['group']=ans.values.copy()
    df.loc[df['Time'] == t, 'group']=ans.values.copy()
print(df)

