#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script works on a set of glass state files defined in `data.txt` and:

1. downloads the corresponding state files

2. derrives a bunch of data from the state file and saves them in a json
data file. If the data file already exists, if ONLY reads the data file!

3. Creates plots with the obtained data

it does this using multiple cores. At least this was the initial intention..


authors:
- Rafael Kueng <rafi.kueng@gmx.ch>

version:
- 2015.??.??  Initial version
- 2018.02.13  clean up and update docu

"""



from __future__ import ( division, absolute_import, print_function, unicode_literals )

import numpy as np
from numpy import zeros
import matplotlib as mpl
import matplotlib.pyplot as plt

import requests as rq
import os
import sys
import json
import zipfile
import multiprocessing as mp
from multiprocessing import Process, Lock


glass_basis('glass.basis.pixels', solver=None)
exclude_all_priors()


outpdir = "kappaencl"
statesdir = os.path.join(outpdir, "states")
imagesdir = os.path.join(outpdir, "images")
datadir = os.path.join(outpdir, "data")


models = [6915]

def work(lock, tasknr, procnr, nr):
    pid = mp.current_process().pid
    with lock:
        print('> %03i (%i; %5i): start %06i' % (tasknr, procnr, pid, nr))


    url = "http://mite.physik.uzh.ch/result/%06i/state.txt" % nr
    fpath = os.path.join(statesdir, "%06i.state"%nr)
    imgfile = os.path.join(imagesdir, "%06i_kappa_encl.png"%nr)
    datafile = os.path.join(datadir, '%06i.json'%nr)
    
    if os.path.isfile(fpath):
        print('> %03i (%i; %5i): skip %06i' % (tasknr, procnr, pid, nr))
        return
    
    if not os.path.isfile(fpath):
        with open(fpath, 'wb') as handle:
            with lock:
                print('> %03i (%i; %5i): start dl %06i' % (tasknr, procnr, pid, nr))

            resp = rq.get(url, stream=True)
        
            if not resp.ok:
                with lock:
                    print('> %03i (%i; %5i): dl failed %06i !!!!!' % (tasknr, procnr, pid, nr))
                return
        
            for block in resp.iter_content(1024):
                if not block:
                    break
        
                handle.write(block)    

            with lock:
                print('> %03i (%i; %5i): dl ok %06i' % (tasknr, procnr, pid, nr))

    with lock:
        print('> %03i (%i; %5i): gen data %06i' % (tasknr, procnr, pid, nr))

    state = loadstate(fpath)

    if os.path.isfile(datafile):
        with open(datafile) as f:
            data = json.load(f)
    else:
        data = gendata(state)
        with open(datafile, 'w') as outfile:
            json.dump(data, outfile)
            
    with lock:
        print('> %03i (%i; %5i): start plot %06i' % (tasknr, procnr, pid, nr))
    plotdata(data, imgfile)

#    gls.make_ensemble_average()
#    gls.kappa_enclosed_plot(gls.ensemble_average)
#    pl.savefig(imgfile)
#    pl.close()

    with lock:
        print('> %03i (%i; %5i): finished %06i' % (tasknr, procnr, pid, nr))

    

#
# copied from simanalysis paper / spaghetti / gen kappa encl data.py
# 
def gendata(state):
    

  distance_factor = 0.428
  div_scale_factors = 440./500*100

    
  state.make_ensemble_average()
  obj,data=state.ensemble_average['obj,data'][0]
  
  n_rings = len(obj.basis.rings) # number of rings with center (=pixrad+1)
  
  #print n_rings
  
  kappaRenc_median = np.zeros(n_rings) #pixrad
#  kappaRd_encl = np.zeros(n_rings) #pixrad
  kappaRenc_1sigmaplus = np.zeros(n_rings) #pixrad
  kappaRenc_1sigmaminus = np.zeros(n_rings) #pixrad
  kappaRd_maxdevplus = np.zeros(n_rings) #pixrad
  kappaRd_maxdevminus = np.zeros(n_rings) #pixrad
  
  pixPerRing = np.zeros(n_rings)
  pixEnc = np.zeros(n_rings)
  
  for i in range(n_rings):
    pixEnc[i] = len(obj.basis.rings[i])
    pixPerRing[i] = len(obj.basis.rings[i])
    for j in range(i):
      pixEnc[i] += len(obj.basis.rings[j])
  
  for k in range(n_rings): #pixrad
    kappaRenc_k_all = np.zeros(0)
    for m in state.models:
      obj,ps = m['obj,data'][0]

      kappaRenc_model = ps['kappa(R)'][k]*pixPerRing[k]
      for kk in range(k):
        kappaRenc_model += ps['kappa(R)'][kk] * pixPerRing[kk]
      kappaRenc_k_all = np.append(kappaRenc_k_all,kappaRenc_model)

    kappaRenc_k_all /= pixEnc[k]
    kappaRenc_k_all *= distance_factor
    
    kappaRenc_k_all = np.sort(kappaRenc_k_all)
    #print kappaRenc_k_all
    #print len(kappaRenc_k_all), pixEnc[k]
    
    kappaRenc_median[k] = kappaRenc_k_all[len(kappaRenc_k_all)/2]
#    kappaRenc_1sigmaplus[k] = kappaRenc_k_all[5*len(kappaRenc_k_all)/6]
#    kappaRenc_1sigmaminus[k] = kappaRenc_k_all[len(kappaRenc_k_all)/6]
    
#    p = 0.0
#    kappaRenc_1sigmaplus[k] = kappaRenc_k_all[int((1.0-p)*len(kappaRenc_k_all))]
#    kappaRenc_1sigmaminus[k] = kappaRenc_k_all[int(p*len(kappaRenc_k_all))]
    kappaRenc_1sigmaplus[k] = kappaRenc_k_all[-1]
    kappaRenc_1sigmaminus[k] = kappaRenc_k_all[0]
    
    #print k, kappaRenc_median[k], kappaRenc_1sigmaplus[k], kappaRenc_1sigmaminus[k]


  pixelradius = n_rings -1

  kappaRd_median = zeros(pixelradius+1) #pixrad
  kappaRd_1sigmaplus = zeros(pixelradius+1) #pixrad
  kappaRd_1sigmaminus = zeros(pixelradius+1) #pixrad
  kappaRd_maxdevplus = zeros(pixelradius+1) #pixrad
  kappaRd_maxdevminus = zeros(pixelradius+1) #pixrad
  
  for k in range(pixelradius+1): #pixrad
    kappaRd_k_all = zeros(0)
    for m in state.models:
      obj,ps = m['obj,data'][0]
      kappaRd_model = ps['kappa(R)'][k]#-kappaRs[k]
      kappaRd_k_all = np.append(kappaRd_k_all,kappaRd_model)
    kappaRd_k_all = np.sort(kappaRd_k_all)
    kappaRd_median[k] = kappaRd_k_all[len(kappaRd_k_all)/2]
    kappaRd_1sigmaplus[k] = kappaRd_k_all[5*len(kappaRd_k_all)/6]
    kappaRd_1sigmaminus[k] = kappaRd_k_all[len(kappaRd_k_all)/6]
    kappaRd_maxdevplus[k] = kappaRd_k_all[-1]
    kappaRd_maxdevminus[k] = kappaRd_k_all[0]


  
  #pl.plot(np.arange(n_rings), kappaRd_median)
#  yerr=[kappaRenc_1sigmaplus - kappaRenc_median, kappaRenc_median- kappaRenc_1sigmaminus]
  
  x_vals = (np.arange(n_rings)+0.5) * div_scale_factors * obj.basis.cell_size[0]
  
  #pl.errorbar(x_vals, kappaRenc_median, yerr=yerr)
  #pl.show()



#  yerr=[kappaRd_1sigmaplus - kappaRd_median, kappaRd_median- kappaRd_1sigmaminus]  

  return {
      'x': x_vals.tolist(),
      'y': kappaRenc_median.tolist(),
      'yp': kappaRenc_1sigmaplus.tolist(),
      'ym': kappaRenc_1sigmaminus.tolist(),
#      'yerr': yerr
  }
  




def plotdata(data, imgname):
    #x_vals, kappaRenc_median, kappaRenc_1sigmaplus, kappaRenc_1sigmaminus, yerr = data
    
    xx = data['x']
    y = data['y']    
    yp = data['yp']    
    ym = data['ym']    
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    x = np.arange(len(xx))
    
    plt.plot(x, yp, 'b--')
    plt.plot(x, ym, 'b--')
    plt.plot(x, y, 'r')
    
    plt.plot([0.01,np.max(x)], [1,1], 'k:')  

    
    plt.tick_params(axis='both', which='both', labelsize=16)


    #plt.tight_layout()
    plt.ylim([0.5,10])
    ax.set_yscale('log')    

    plt.xlabel(r'image radius [pixels]', fontsize = 18)
    plt.ylabel(r'mean convergance [1]', fontsize = 18)

    formatter = mpl.ticker.FuncFormatter(lambda x, p: '$'+str(int(round(x)))+'$' if x>=1 else '$'+str(round(x,1))+'$')
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)

    #plt.show()
    plt.savefig(imgname)



def main():
    models = []
    with open('data.txt') as f:
        for line in f:
            models.append(int(line))
            
    print(models)
    rmodels = list(reversed(models))
        
    
    procs = [None, None, None, None]
    tasknr = 0
    l = Lock()
    
#    while len(models) > 0:
#        
#        for procnr, p in enumerate(procs):
#            if p:
#                p.join(1)
#                if not p.is_alive():
#                    p = None
#                
#            if not p and len(models)>0:
#                tasknr += 1
#                nr = models.pop()
#
#                with l:
#                    print("XXX   (X; XXXXX): start p%03i on %i (%06i)" % (tasknr, procnr, nr))
#                    sys.stdout.flush()
#                p = Process(target=work, args=(l, tasknr, procnr, nr,))
#                p.start()

    while len(models) > 0:
        tasknr += 1
        nr = rmodels.pop()
        try:
            work(l, tasknr, 0, nr )        
        except zipfile.BadZipfile:
            pass


                
#
#        
#    my_list = range(1000000)
#
#    q = Queue()
#
#    p1 = Process(target=do_sum, args=(q,my_list[:500000]))
#    p2 = Process(target=do_sum, args=(q,my_list[500000:]))
#    p1.start()
#    p2.start()
#    r1 = q.get()
#    r2 = q.get()
#    print r1+r2

if __name__=='__main__':
    main()



#
#opts = Environment.global_opts['argv']
#
#
#sl_nrs = [int(_) for _ in opts[1:]]
#
#print( 'opts:', opts)
#print( 'sl_nrs:', sl_nrs)
#
#
#statedir = 'state'
#plotdir = 'plots'
#
#try:
#    os.mkdir(statedir)
#except OSError:
#    pass
#try:
#    os.mkdir(plotdir)
#except OSError:
#    pass
#
#
#for nr in sl_nrs:
#    
#    print('Working on %06i'%nr)
#    print('  > fetching state file ...', end='')
#
#    url = "http://mite.physik.uzh.ch/result/%06i/state.txt" % nr
#    statefilename = os.path.join(statedir, '%06i.state' % nr)
#    imgfilename1 = os.path.join(plotdir, '%06i_dt_plot_f1.png' % nr)
#    imgfilename25 = os.path.join(plotdir, '%06i_dt_plot_f25.png' % nr)
#    
#    with open(statefilename, 'wb') as handle:
#        response = requests.get(url, stream=True)
#    
#        if not response.ok:
#            print('FAIL. skipping..')
#            continue
#    
#        for block in response.iter_content(1024):
#            if not block:
#                break
#    
#            handle.write(block)    
#        
#        print('OK')
#
#    print('  > plotting ...', end='')
#    gls = loadstate(statefilename)
#    gls.time_delays_plot(arb_fact = 2.50)
#    pl.savefig(imgfilename25)
#    pl.close()
#    gls.time_delays_plot(arb_fact = 1)
#    pl.savefig(imgfilename1)
#    pl.close()
#    print('OK')


