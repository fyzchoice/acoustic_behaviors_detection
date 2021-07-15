import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import socket
import uuid
import time
import os
import threading
import logging
import struct
from logging import handlers
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inference import Array
import tensorflow as tf
# from VideoCapture import Device
# 获取MAC地址
def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])


# 获取IP地址
def get_host_ip():
    try:
        my = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        my.connect(('8.8.8.8', 80))
        # ip = my.getsockname()[0]
        ipList = my.getsockname()
    finally:
        my.close()
    return ipList


def _logging(**kwargs):
    level = kwargs.pop('level', None)
    filename = kwargs.pop('filename', None)
    datefmt = kwargs.pop('datefmt', None)
    format = kwargs.pop('format', None)
    if level is None:
        level = logging.DEBUG
    if filename is None:
        filename = 'default.log'
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S'
    if format is None:
        format = '%(asctime)s [%(module)s] %(levelname)s [%(lineno)d] %(message)s'

    log = logging.getLogger(filename)
    format_str = logging.Formatter(format, datefmt)
    # backupCount 保存日志的数量，过期自动删除
    # when 按什么日期格式切分(这里方便测试使用的秒)
    th = handlers.TimedRotatingFileHandler(filename=filename, when='H', backupCount=3, encoding='utf-8')
    th.setFormatter(format_str)
    th.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    #  ch.setFormatter(format)
    log.addHandler(ch)

    log.addHandler(th)
    log.setLevel(level)
    return log


''' 
def video(udpServerSocket,addr):

    # 需要传输的文件路径
    filepath = '/home/ys/Pictures/1.jpg'
    # 判断是否为文件
    if os.path.isfile(filepath):
        # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
        fileinfo_size = struct.calcsize('128sl')
        # 定义文件头信息，包含文件名和文件大小
        fhead = struct.pack('128sl', os.path.basename(filepath).encode('utf-8'), os.stat(filepath).st_size)
        # 发送文件名称与文件大小
        udpServerSocket.sendto(fhead.encode('utf-8'), addr)

        # 将传输文件以二进制的形式分多次上传至服务器
        fp = open(filepath, 'rb')
        while 1:
            data = fp.read(1024)
            if not data:
                print ('{0} file send over...'.format(os.path.basename(filepath)))
                break
            udpServerSocket.sendto(data.encode('utf-8'), addr)
        # 关闭当期的套接字对象
        udpServerSocket.close()
'''
'''
 while True:
            print(self.recvData)
            if self.recvData == b'1':
                ret, fra = cap.read()
                if ret:
                    print(ret)
                    _, sendData = cv2.imencode('.jpg', fra)
                    print(sendData.size)
                    print(fra.size)
                    self.sendSocket.sendto(sendData, ('255.255.255.255', self.sendbroadcastPort))

            sleep(1)
        self.sendSocket.close()
        cap.release()
'''


def video(addr):
    # 读取图片，编码成二进制 bytes格式
    cap = cv2.VideoCapture(0)
    '''
    cap.set(3,160)
    cap.set(4,160)'''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1000, 700))
        pictureData = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
        sYS.sendto(pictureData, (re[1][0], 9999))
    sYS.close()
    cap.release()


command = ['停止', '前进', '后退', '左移', '右移', '左转', '右转', '关机', '重启', '重新启动']




os.makedirs("../logs", exist_ok=True)
mylog = _logging(filename='../logs/udpserver.log')

print("等待3秒")
mylog.debug("等待3秒")
time.sleep(3)
print("等待结束")
mylog.debug("等待结束")

HOST = ''
PORT = 8888
BUFSIZ = 2048
ADDRESS = (HOST, PORT)

udpServerSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpServerSocket.bind(ADDRESS)  # 绑定客户端口和地址
sYS = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
re = udpServerSocket.recvfrom(2048)
myname = socket.gethostname()
print("myname:" + myname)
mylog.debug("myname:" + myname)
myIPList = get_host_ip()
print("myIPList:" + str(myIPList))
mylog.debug("myIPList:" + str(myIPList))
macAddress = get_mac_address()
print("macAddress:" + macAddress)
mylog.debug("macAddress:" + macAddress)

toClose = False
data, addr = udpServerSocket.recvfrom(BUFSIZ)

#使用GPU,动态申请GPU内存
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')

model=tf.keras.models.load_model('../Models/emg_model.h5')
alist=list()
print(alist)
label=['向前抓取', '拾取', '猛打方向盘', '转身','正常驾驶']
def main():
    global udpServerSocket
    print("enslfjslfjwlfjwlfjwlkfmwlkefnwlkfnewlkn")
    global toClose
    count=0
    while True:
        data, addr = udpServerSocket.recvfrom(BUFSIZ)
        # print("waiting for message...")
        # mylog.debug("waiting for message...")

        currCode = data.decode('utf-8')
        # print("接收到数据：" + currCode)
        # mylog.debug("接收到数据：" + currCode)
        if(len(alist)<160):
            print('等待开始')
            alist.append(float(currCode))
        elif(len(alist)>=160):
            alist.pop(0)
            alist.append(float(currCode))
            count=count+1
            # print("count:",count)
            #实时推理
            if(count%5==0):
                data = np.array(alist)
                plt.clf()  # 清除之前画的图
                plt.plot(data)
                plt.pause(0.02)
                plt.ioff()  # 关闭画图窗口
                data = np.reshape(data, (1,-1, 40))
                pre = model.predict(data)

                if(np.max(pre)<0.6):
                    print(label[4])
                else:
                    labelid = np.argmax(pre)
                    print(pre)
                    print(label[labelid])
                    count=0

        # print(len(alist))

        # plt.clf() # 清除之前画的图
        # plt.plot(alist)
        #
        # plt.pause(0.1) # 暂停一秒
        # plt.ioff()# 关闭画图的窗口


        # content = '[%s] %s' % (bytes(ctime(), 'utf-8'), data.decode('utf-8'))
        # # 发送服务器时间
        # if currCode == "TIME":
        #     content = "Time:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        # # 发送IP地址
        # elif currCode == "IP":
        #     content = "IP:" + str(myIPList)
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        # # 发送mac地址
        # elif currCode == "MAC":
        #     content = "MAC:" + macAddress
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        # # 发送ip mac地址
        # elif currCode == "IP_MAC":
        #     content = "IP:" + str(myIPList) + "|MAC:" + macAddress
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        # # 退出UDP服务端
        # elif currCode == "EXIT":
        #     content = "程序中止"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print(content)
        #     toClose = True
        # # 重启
        # elif currCode == "REBOOT":
        #     content = "服务端重启"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("服务端开始重启")
        #     mylog.debug("服务端开始重启")
        #     # os.system('reboot')
        #     break
        # # 关机
        # elif currCode == "SHUTDOWN":
        #     content = "关机"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("关机")
        #     mylog.debug("关机")
        #     # os.system('sudo shutdown -h now')
        #     # os.system('python3 video.py')
        #     break
        # elif currCode == "MOVELEFT":
        #     content = "MOVELEFT"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("左移")
        #     mylog.debug("左移")
        # elif currCode == "MOVERIGHT":
        #     content = "MOVERIGHT"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("右移")
        #     mylog.debug("右移")
        # elif currCode == "TURNLEFT":
        #     content = "TURNLEFT"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("左转")
        #     mylog.debug("左转")
        # elif currCode == "TURNRIGHT":
        #     content = "TURNRIGHT"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("右转")
        #     mylog.debug("右转")
        # elif currCode == "FORWARD":
        #     content = "FORWARD"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("前进")
        #     mylog.debug("前进")
        # elif currCode == "BACK":
        #     content = "BACK"
        #     udpServerSocket.sendto(content.encode('utf-8'), addr)
        #     print("后退")
        #     mylog.debug("后退")
        # else:
        #     udpServerSocket.sendto("Bad Key".encode('utf-8'), addr)
        #
        # # content = '[%s] %s %s' % (bytes(ctime(), 'utf-8'), str(myIPList), macAddress)
        # # udpServerSocket.sendto(content.encode('utf-8'), addr)
        # print('...received from and returned to:', addr)
        # # mylog.debug('...received from and returned to:', addr)
    udpServerSocket.close()
    print("服务端退出")
    mylog.debug('服务端退出')


def go(order):
    if order == 'yes':
        main()
    else:
        video(addr)


list = ['yes', 'audio', 'no']

threads = []
files = range(len(list))

# 创建线程
for i in files:
    t = threading.Thread(target=go, args=(list[i],))
    threads.append(t)

if __name__ == '__main__':
    for i in files:
        threads[i].start()