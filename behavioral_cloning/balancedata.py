from pathlib import Path
from pandas import read_csv
from sys import exit
from tqdm import trange
from random import shuffle

header = 'center,left,right,steering,throttle,brake,speed'
drivinglog = './driving_log.csv'
outlog = './driving_log.csv'
numbns = 10

# save cvs file with the selected images per bin
def gen_csv(outlog, df, bins):
    outdata = []
    minlen = 99999
    # find angle with less images in dataset
    for k in bins:
        ln = len(bins[k])
        print('Total images in bin [{:>3d}]: {:>5d}, ej.: {:2.4f}'.format(k, ln, df['steering'][bins[k][0]]))
        if ln < minlen:
            minlen = ln
    for k in bins:
        # randomly shuffle the images
        shuffle(bins[k])
        shuffle(bins[k])
        # select same amount of images from each bin.
        for i in range(minlen):
            outdata.append(bins[k][i])

    print('Min number of images: {}'.format(minlen))

    try:
        fcsv = open(outlog, 'w')
    except OSError as e:
        print('Error generating out driving log {}'.format(e))
        return

    fcsv.write(header+'\n')

    for i in outdata:
        line = df['center'][i] + ',' \
                + str(df['left'][i]) + ',' \
                + str(df['right'][i]) +',' \
                + str(df['steering'][i]) + ',' \
                + str(df['throttle'][i]) + ',' \
                + str(df['brake'][i]) + ',' \
                + str(df['speed'][i]) + '\n'
        fcsv.write(line)
    fcsv.close()
    return




if __name__=='__main__':

    df = read_csv(drivinglog)
    bins = {}
    # classify images in each bin
    for i in trange(len(df['center'])):
        steer = df['steering'][i]
        bn = int(steer * numbns)
        if bn == 0: bn = 1
        if bn not in bins:
            bins[bn] = [i]
        else: bins[bn].append(i)

    gen_csv(outlog, df, bins)
