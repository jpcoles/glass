class Commands:
    glass_command_list = {}
    _env = None

    @classmethod
    def set_env(self, env):
        self._env = env

    @classmethod
    def get_env(self):
        return self._env

    def __str__(self):
        return '\n'.join([n for n in glass_command_list.items()])

def command(f):
    def g(*args, **kwargs):
        #print 'calling', f.__name__, env(), args, kwargs
        return f(Commands.get_env(), *args, **kwargs)
    Commands.glass_command_list[f.__name__] = [f,g]
    #print 'Creating command', f.__name__
    #print glass_command_list
    return g


