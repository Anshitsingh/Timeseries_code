import requests
from bs4 import BeautifulSoup 
import re
import numpy as np
import pandas as pd
url="Enter the url for wind "
r=requests.get(url)
html=r.content
a=html.split()
p=1
columns=['Forecast/Hour','TT(Deg.C)','Td(Deg.C)','RH(%)','WD(10m)(Deg.)','WS(10m)(Deg.)','Rain(mm)']
x= ((len(a)+1) /7)
df = pd.DataFrame(np.zeros(shape=(x,7 ),columns=columns))
p=11
k=0
df.set_index('Forecast/Hour',inplace=True)
df.index = pd.to_datetime(df.index)
while (p < (len(a)+1)):
                        j=0
                        while (j<7):
                                    df[k][j]=a[p]
                                    j=j+1
                        k=k+1
i=j=0
for i in range(x):
	for j in range(7):
		print df[i][j]
		print("\n")
