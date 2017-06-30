from ftplib import FTP
import socket
ftp = FTP('1**.*.*...') 
ftp.login(user='enteruserid',passwd='enterpassword')
ftp.cwd('fcst')
i=4
while(i<7):
        j=1
        while(j<10):
            fileName='dadfrcst0'+str(j)+'0'+str(i)+'r2.xlsx'
            filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data0'+str(j)+'0'+str(i)+'.xlsx'
            localfile=open(filename,'wb')
            ftp.retrbinary('RETR '+fileName,localfile.write,1024)
            print 'Written file_data0'+str(j)+'0'+str(i)+'.xlsx'
            j=j+1
            localfile.close()
        
        if(i==5):
            j=10
            while(j<32):
                fileName='dadfrcst'+str(j)+'0'+str(i)+'r2.xlsx'
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data'+str(j)+'0'+str(i)+'.xlsx'
                localfile=open(filename,'wb')
                ftp.retrbinary('RETR '+fileName,localfile.write,1024)
                print 'Written file_data'+str(j)+'0'+str(i)+'.xlsx'
                localfile.close()
                j=j+1
            print "\n"
        if(i==4):
            j=10
            while(j<31):
                fileName='dadfrcst'+str(j)+'0'+str(i)+'r2.xlsx'
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data'+str(j)+'0'+str(i)+'.xlsx'
                localfile=open(filename,'wb')
                ftp.retrbinary('RETR '+fileName,localfile.write,1024)
                print 'Written file_data'+str(j)+'0'+str(i)+'.xlsx'
                localfile.close()
                j=j+1
            print "\n"
        if(i==6):
            j=10
            while(j<28):
                fileName='dadfrcst'+str(j)+'0'+str(i)+'r2.xlsx'
                filename='C:\Users\user\Desktop\INTERNSHIP\SOLAR_DADRI\DATA\data'+str(j)+'0'+str(i)+'.xlsx'
                localfile=open(filename,'wb')
                ftp.retrbinary('RETR '+fileName,localfile.write,1024)
                print 'Written file_data'+str(j)+'0'+str(i)+'.xlsx'
                localfile.close()
                j=j+1
            print "\n"
        i=i+1
ftp.close()