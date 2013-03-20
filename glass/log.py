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
