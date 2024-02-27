import os, torch

def collect_data(d):
    data = []
    for i in os.listdir(d):
        with open(f'{d}/{i}') as f:
            lines = f.readlines()
            key = ([int(i) for i in lines[0].strip().split(',') if len(i) > 0], float(lines[1].strip()), float(lines[2].strip()))
            data += [key]
    return data

def key_diff(k0, k1):
    diffarr = [abs(i-j) for i, j in zip(k0, k1)]
    return sum(diffarr), diffarr


def compare(d0, d1):
    data0 = collect_data(d0)
    data1 = collect_data(d1)
    mindiff = 100000000000000
    maxdiff = 0
    cnt0 = 0
    for dt0 in data0:
        key0, x0, y0 = dt0
        for dt1 in data1:
            key1, x1, y1 = dt1
            assert sum(key0) == sum(key1)
            diff_key, diffarr = key_diff(key0, key1)
            #print(diff_key)
            #if diff_key < 1000000:
            #    import pdb; pdb.set_trace()
            
            #    print()
            if diff_key == 0:
                cnt0 += 1
            if diff_key < mindiff:
                mindiff = diff_key
                middiffarr = diffarr
                minx1 = x1
                miny1 = y1
            if diff_key > maxdiff:
                maxdiff = diff_key
        print(mindiff, (minx1-x0)/x0, (miny1-y0)/y0)
        print('-------')
    print(mindiff, maxdiff, cnt0)

import numpy as np
from PIL import Image as im     
def compare_tensor(d0, d1):
    try:
        os.mkdir('compare')
    except:
        pass
    d0data = [torch.load(f'{d0}/{tensor}') for tensor in os.listdir(d0)]
    d1data = [torch.load(f'{d1}/{tensor}') for tensor in os.listdir(d1)]
    matchcnt = 0
    for idx1, (t01, t11, t21) in enumerate(d1data):
        t01_flipped0 = torch.flip(t01, [2])
        #import pdb; pdb.set_trace()
        mindiff = 1000000000000
        whichdiff = ''
        for idx0, (t00, t10, t20) in enumerate(d0data):
            diff1 = torch.abs(t01 - t00).sum()
            diff2 = torch.abs(t01_flipped0 - t00).sum()
            if diff1 < diff2:
                whichdiff = 'smallest is diff1'
                diff = diff1
                t01_curr = t01
            else:
                whichdiff = 'smallest is diff2'
                diff = diff2
                t01_curr = t01_flipped0

            if diff < mindiff:
                bestmatch = (t01_curr, t11, t21, t00, t10, t20, whichdiff)
                mindiff = diff
                bestidx = idx0
                #print(bestidx)
        print(idx1, bestidx,  bestmatch[6])
        i1 = (bestmatch[0].permute(1,2,0).numpy() + 1) / 2
        i0 = (bestmatch[3].permute(1,2,0).numpy() + 1)/2
        #import pdb; pdb.set_trace()
        concatted = (np.concatenate((i0,i1)) * 255).astype(np.uint8)
        pilimg = im.fromarray(concatted)
        import pdb; pdb.set_trace()
        pilimg.save(f'compare/{idx1}.png')
        #if idx1 == 9 or idx1 == 8:
        #    import pdb; pdb.set_trace() 
        #    print()
          

#compare('dump1_mediapipe', 'dump2_nomediapipe')
#compare('dump1_mediapipe', 'dump1_mediapipe')
#compare('dump4_mediapipe', 'dump2_nomediapipe')

#compare_tensor('dump7_tensor_mediapipe', 'dump8_tensor_nomediapipe')

compare_tensor('dump8_tensor_nomediapipe', 'dump8_tensor_nomediapipe')