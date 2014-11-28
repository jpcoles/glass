from __future__ import ( division, absolute_import, print_function, unicode_literals )

#import matplotlib.pylab as pl
import matplotlib.pyplot as pl
import requests
import os


glass_basis('glass.basis.pixels', solver=None)
exclude_all_priors()


opts = Environment.global_opts['argv']


sl_nrs = [int(_) for _ in opts[1:]]

print( 'opts:', opts)
print( 'sl_nrs:', sl_nrs)

statedir = 'state'
plotdir = 'plots'

try:
    os.mkdir(statedir)
except OSError:
    pass
try:
    os.mkdir(plotdir)
except OSError:
    pass


for nr in sl_nrs:
    
    print('Working on %06i'%nr)
    print('  > fetching state file ...', end='')

    url = "http://mite.physik.uzh.ch/result/%06i/state.txt" % nr
    statefilename = os.path.join(statedir, '%06i.state' % nr)
    imgfilename1 = os.path.join(plotdir, '%06i_dt_plot_f1.png' % nr)
    imgfilename25 = os.path.join(plotdir, '%06i_dt_plot_f25.png' % nr)
    
    with open(statefilename, 'wb') as handle:
        response = requests.get(url, stream=True)
    
        if not response.ok:
            print('FAIL. skipping..')
            continue
    
        for block in response.iter_content(1024):
            if not block:
                break
    
            handle.write(block)    
        
        print('OK')

    print('  > plotting ...', end='')
    gls = loadstate(statefilename)
    gls.time_delays_plot(arb_fact = 2.50)
    pl.savefig(imgfilename25)
    pl.close()
    gls.time_delays_plot(arb_fact = 1)
    pl.savefig(imgfilename1)
    pl.close()
    print('OK')


