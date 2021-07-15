import os
path = 'emg/'

i=1

k=0
files=os.listdir(path)

for m in range(files.__len__()-1):
    # if(m%10<10):
        file=files[m]
        if os.path.isfile(os.path.join(path,file))==True:
            new_name = file.replace(file,"sky"+str(i)+".csv")
            os.rename(os.path.join(path,file),os.path.join(path,new_name))
            i=i+1
