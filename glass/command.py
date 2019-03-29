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
        return '\n'.join([n for n in list(glass_command_list.items())])

def command(*args):

    if isinstance(args[0], str):
        help_text = args[0]
        def h(f):
            def g(*args, **kwargs):
                return f(Commands.get_env(), *args, **kwargs)
            Commands.glass_command_list[f.__name__] = [f,g,help_text]
            return g
        return h
    else:
        f = args[0]
        help_text = ''
        def g(*args, **kwargs):
            return f(Commands.get_env(), *args, **kwargs)
        Commands.glass_command_list[f.__name__] = [f,g,help_text]
        return g



