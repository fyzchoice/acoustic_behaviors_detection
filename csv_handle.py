import os
import re
path = 'D:\\python_program\\csv_handle\\csv\\'
fileheadpre='pre_'
fileheadafter='after_'
i=1
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file))==True:
        new_name = file.replace(file,fileheadpre+str(i)+".csv")
        os.rename(os.path.join(path,file),os.path.join(path,new_name))
        i+=1
print("数据预处理完成\n")
print("是否要删除数据，输入Y或者N\n")
judge = input("ENTER Y OR N\n")
if str(judge)=='Y' or str(judge)=='y':
    while True:
        print('请输入要删除的序号,或输入9999退出\n')
        NO = input("enter a number to be delete\n")
        if int(NO)==9999:
            break
        else:
            delete_file_path = path+fileheadpre+str(NO)+'.csv'
            os.remove(delete_file_path)
            print("成功删除"+str(NO)+"号文件")
i=1
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file))==True:
        new_name = file.replace(file,fileheadafter+str(i)+".csv")
        os.rename(os.path.join(path,file),os.path.join(path,new_name))
        i+=1
files = os.listdir(path)
print("文件删除结束\n")
epoch = 10
for total in range(6):
    head_now = str(input("请输入第"+str(total)+'组的100个数据要改的名字\n'))
    for once in range(epoch):
        current_csv = (total)*epoch+once+1
        path_now = path+fileheadafter+str(current_csv)+".csv"
        if os.path.isfile(os.path.join(path,files[current_csv]))==True:
            new_name = files[current_csv].replace(files[current_csv],head_now+str(once)+".csv")
            os.rename(os.path.join(path,files[current_csv]),os.path.join(path,new_name))
print('end')