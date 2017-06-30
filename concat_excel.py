import pandas as pd
from datetime import datetime
a=pd.read_excel("C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data0106.xlsx",Header='None')
a=a[1:]
del a['Unnamed: 0']
del a['Unnamed: 1']

a.columns=['Date','Values']

i=6
while(i<7):
        j=2
        while(j<10):
            filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data0'+str(j)+'0'+str(i)+'.xlsx'
            j=j+1
            b=pd.read_excel(filename,Header='None')
            b=b[1:]
            del b['Unnamed: 0']
            del b['Unnamed: 1']
            b.columns=['Date','Values']
            a=pd.concat([a,b],axis=0)
        if(i==5):
            while(j<32):
                            
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data0'+str(j)+'0'+str(i)+'.xlsx'
                j=j+1
                b=pd.read_excel(filename,Header='None')
                b=b[1:]
                del b['Unnamed: 0']
                del b['Unnamed: 1']
                b.columns=['Date','Values']
                a=pd.concat([a,b],axis=0)
        if(i==4):
            
            while(j<31):
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data0'+str(j)+'0'+str(i)+'.xlsx'
                j=j+1
                b=pd.read_excel(filename,Header='None')
                b=b[1:]
                del b['Unnamed: 0']
                del b['Unnamed: 1']
                b.columns=['Date','Values']
                pd.concat([x,b],axis=0)
        if(i==6):
            while(j<28):
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data'+str(j)+'0'+str(i)+'.xlsx'
                j=j+1
                b=pd.read_excel(filename,Header='None')
                b=b[1:]
                del b['Unnamed: 0']
                del b['Unnamed: 1']
                b.columns=['Date','Values']
                a=pd.concat([a,b],axis=0)
        i=i+1

a['Date']=pd.to_datetime(a['Date'])
a
writer=pd.ExcelWriter("C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\COMBINED\06.xlsx")
a.to_excel(writer,'Sheet1')
writer.save()