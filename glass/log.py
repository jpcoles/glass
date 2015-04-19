import sys
from command import command

log_files = [[sys.stdout, False]]

@command
def setup_log(env, *files, **kwargs):
    global log_files
    stdout = kwargs.get('stdout', True)
    stderr = kwargs.get('stderr', False)
    log_files = [ [open(f,'a'),False] if isinstance(f,str) else [f,False] for f in files ]
    if stdout: log_files.append([sys.stdout, False])
    if stderr: log_files.append([sys.stderr, False])

def log(*args, **kwargs):
    overwritable = kwargs.get('overwritable', False)
    for fo in log_files:
        f,is_overwritable = fo
        s = '\r' if f.isatty() else ''
        if is_overwritable and f.isatty():
            s += '\x1b[2K'
        s += '\n    '.join(args)
        if not overwritable or not f.isatty():
            s += '\n'
        f.write(s)
        f.flush()
        fo[1] = overwritable


status_reporter = []
status_history = []

@command
def setup_status_reporter(env, funcs):
    global status_reporter
    status_reporter = funcs
    
@command
def get_status_history(env):
    return status_history
    
@command
def set_status(env, *args, **kwargs):
    Status(*args, **kwargs)

    
def Status(*args, **kwargs):
    # gets called each time the status of glass changes
    # has information about what glass is currently doing
    # and progress states
    # in "machine readable format"
    # use this from other progs instead of parsing the log file or stdout..
    # calles all registered functions (use: setup_status_report)
    # upon status change
    # and passes the complete history

    # use past for single events with no progress state ("started bla")
    # and present cont. for ongoing events ("calculating EV")

    # how to call inside glass:
    # Status("blabla")
    # Status("blabla", i=0, of=100)
    #
    # kwargs:
    # text:     str, what is it doing at the moment
    # i:        whats done (int)
    # of:       whats to be done (int)
###    # progress: (i, n) integers representing the progress "i of n done"

    # the foreign function will be passed the whole status_history
    # 

    global status_history
    
    if not status_reporter:
        return
    
    if len(args)==3 and len(kwargs)==0:
        text = args[0]
        i = int(args[1])
        n = int(args[2])
    elif len(args)==0:
        text = kwargs.get('text', "--missing--")
        i = int(kwargs.get('i', -1))
        n = int(kwargs.get('of', -1))
    else:
        text = " | ".join(args)
        i = 0
        n = 0
    
    try:
        oldtxt = status_history[-1][0]
    except:
        oldtxt = None
        
    if oldtxt and oldtxt == text: # the last entry has the same text: only update the status
        status_history[-1][1] = i
        status_history[-1][2] = n
    else: # append a new status
        status_history.append([text, i, n])

    # call all the registered reporting functions
    for reporter in status_reporter:
        try:
            reporter(status_history)
        except:
            # don't crash if some foeign function crashes
            pass


