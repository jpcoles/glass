import sys

log_files = [sys.stdout]

def setup_log(files=[], stdout=True, stderr=False):
    global log_files
    if not hasattr(files, '__iter__'): files = [files]
    log_files = [ open(f,'a') if isinstance(f,str) else f for f in files ]
    if stdout: log_files.append(sys.stdout)
    if stderr: log_files.append(sys.stderr)

    
def log(s=''):
    for f in log_files:
        print >>f, s
