import daemon, time

def printNumbers():
    for i in range(10000):
        print(i)
        time.sleep(0.1)

if __name__ == '__main__':
    print('Testing daemon...')
    with daemon.DaemonContext():
        printNumbers()
