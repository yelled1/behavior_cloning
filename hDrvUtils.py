"""hlpUtils.py: Hlper scripts for CarND"""
import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

tDir = 'combDir/'
tFnm = 'driving_tst.csv'
mFnm = 'driving_mod.csv'
vFnm = 'driving_val.csv'


def shuffle(df, n=1, axis=0):  #this does both row, col
    df = df.copy()
    for _ in range(n): df.apply(np.random.shuffle, axis=axis)
    return df

def spltLog(dDir=tDir, valPerc=.2, tstPerc=.1):
    dl = pd.read_csv(dDir+'driving_log.csv')
    nl = dl.reindex(np.random.permutation(dl.index)) #shuffles row only
    vx = +int(len(nl)*valPerc)
    tx = -int(len(nl)*tstPerc)
    nl[:vx].to_csv(dDir+vFnm, sep='\t')
    print('Creating', dDir+vFnm, nl[:vx].shape)
    nl[tx:].to_csv(dDir+tFnm, sep='\t')
    nl[vx:tx].to_csv(dDir+mFnm, sep='\t')
    return vx, tx

def incrementFiles(FName):
    """ incrementFiles whenver it find existing """
    if os.path.isfile(FName):
        mvFile = FName+'.'+str(len(glob.glob(FName+'*')))
        try:    os.rename(FName,  mvFile)
        except: raise "File % Exists Error" %mvFile

def saveModel(sDir, model, modelFName='model.json', WgtFName='model.h5'):
    incrementFiles(sDir+modelFName)
    incrementFiles(sDir+WgtFName)
    jsonStr = model.to_json()
    with open(sDir+modelFName, 'w') as fp: 
        json.dump(jsonStr, fp)
    model.save_weights(sDir+WgtFName)

def getImageInBatch(batch_size, dDir=tDir, STEERING_COEF=0.229, logFnm=mFnm):
    """
    returns: An list of selected (image files names, steering angles)
    """
    data    = pd.read_csv(dDir+logFnm, sep='\t').ix[:, 1:] # Need to deal with No Headers
    rnd_ixs = np.random.randint(0, len(data), batch_size)

    lcrDict = { -1: 'left', 0: 'center', 1: 'right'}
    imgs_angles_list = []
    for ix in rnd_ixs:
        rand_choice = np.random.randint(-1, 2)
        img   = data.iloc[ix][lcrDict[rand_choice]].strip()
        angle = data.iloc[ix]['steering'] + rand_choice * STEERING_COEF
        imgs_angles_list.append((img, angle))
    return imgs_angles_list

def reSize(image, nDim=(64,64)): return scipy.misc.imresize(image, nDim)

def generateNewBatch(dDir, batch_size=64, lFnm=tFnm):
    """
    Generator of next training batch
    :param batch_size: # of training images in a single batch
    :return:           A (tuple) of features and steering angles as 2 np arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = getImageInBatch(batch_size, dDir=dDir, logFnm=lFnm)
        for img_file, steer_angle in images:
            raw_image = plt.imread(dDir + img_file)
            raw_angle = steer_angle
            X_batch.append(reSize(raw_image))
            y_batch.append(raw_angle)
        assert len(X_batch) == batch_size,  'len(X_batch) == batch_size Must be True'
        yield np.array(X_batch), np.array(y_batch)

if __name__ == '__main__':
    pass
    #x = generateNewBatch('./data/', batch_size=64)
    #spltLog(dDir=tDir, valPerc=.2, tstPerc=.1)
