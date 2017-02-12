from pathlib import Path
from pandas import read_csv
from sys import exit
from tqdm import trange
from os import remove

drivinglog = './driving_log_all.csv'
header = 'center,left,right,steering,throttle,brake,speed'
outcsv = './driving_log.csv'


if __name__ == '__main__':
    try:
        log = read_csv(drivinglog)
        outf = open(outcsv, 'w')
        outf.write(header+'\n')
    except OSError as e:
        print('Error opening driving log. {}'.format(e))
        exit(-1)
    
    for i in trange(len(log['center'])):
        imgfile = Path(log['center'][i])
        if imgfile.is_file():
            if (log['steering'][i] == 0.0) or (log['steering'][i] >= 0.6) or (log['steering'][i] <= -0.6):
                remove(str(imgfile.absolute()))
            else:
                line = log['center'][i] + ',' \
                    + str(log['left'][i]) + ',' \
                    + str(log['right'][i]) +',' \
                    + str(log['steering'][i]) + ',' \
                    + str(log['throttle'][i]) + ',' \
                    + str(log['brake'][i]) + ',' \
                    + str(log['speed'][i]) + '\n'
                outf.write(line)
    outf.close()



