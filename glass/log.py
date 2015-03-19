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
def setup_status_report(env, funcs):
    global status_reporter
    status_reporter = funcs

    
def status(**kwargs):
    # gets called each time the status of glass changes
    # has information about what glass is currently doing
    # and progress states
    # in "machine readable format"
    # use this from other progs instead of parsing the log file or stdout..
    # calles all registered functions (use: setup_status_report)
    # upon status change
    # and passes the complete history

    # kwargs:
    # text:     str, what is it doing at the moment
    # progress: (i, n) integers representing the progress "i of n done"

    global status_history
    
    if not status_reporter:
        return
    
    text = kwargs.get('text', "--missing--")
    prog = kwargs.get('progress', (-1, -1))
    
    try:
        oldtxt = status_history[-1][0]
    except:
        oldtxt = None
        
    if oldtxt and oldtxt == text: # the last entry has the same text: only update the status
        status_history[-1][1] = prog
    else: # append a new status
        status_history.append([text, prog])

    # call all the registered reporting functions
    for reporter in status_reporter:
        try:
            reporter(status_history)
        except:
            # don't crash if some foeign function crashes
            pass


