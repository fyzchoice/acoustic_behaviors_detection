# import signal
# import threading
#
# import speedtest_cli
#
#
# def testSpeed(urls):
#     speedtest_cli.shutdown_event = threading.Event()
#     signal.signal(signal.SIGINT, speedtest_cli.ctrl_c)
#
#     print("Start to test download speed: ")
#     dlspeed = speedtest_cli.downloadSpeed(urls)
#     dlspeed = (dlspeed / 1000 / 1000)
#     print('Download: %0.2f M%s/s' % (dlspeed, 'B'))
#
#     return dlspeed
#
#
# speed=testSpeed('www.baidu.com')
# print(speed)


# import speedtest
#
# st = speedtest.Speedtest()
# st.get_best_server()
#
# ping = st.results.ping
# download = st.download()
# upload = st.upload()
#
# print('download',download)
# print('upload',upload)


import speedtest

servers = []
# If you want to test against a specific server
# servers = [1234]

threads = None
# If you want to use a single threaded test
# threads = 1

s = speedtest.Speedtest()
# s.get_servers(servers)
s.get_best_server()
s.download(threads=threads)
s.upload(threads=threads)
s.results.share()

results_dict = s.results.dict()

print(results_dict)
print('download:',results_dict['download']/(1024*1024*8),'MB')
print('upload:',results_dict['upload']/(1024*1024*8),'MB')