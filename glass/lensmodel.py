def prop(name):
    def w(func):
        func.lens_model_prop_name = name
        return func
    return w

class LensModel(dict):
    def __init__(self, obj):
        dict.__init__(self)
        self.obj = obj

    def __getitem__(self, item):
        if dict.__contains__(self, item):
            return dict.__getitem__(self, item)

        for m,f in list(self.__class__.__dict__.items()):
            if hasattr(f, 'lens_model_prop_name'):
                if item == f.lens_model_prop_name:
                    return f(self)

        s = item.rsplit(None,1)
        if len(s) == 2:
            for m,f in list(self.__class__.__dict__.items()):
                if hasattr(f, 'lens_model_prop_name'):
                    if s[0] == f.lens_model_prop_name:
                        return f(self, [s[1]])

    def __contains__(self, item):
        if dict.__contains__(self, item):
            return True

        for m,f in list(self.__class__.__dict__.items()):
            if hasattr(f, 'lens_model_prop_name'):
                if item == f.lens_model_prop_name:
                    return True
        return False

    def has_key(self, item):
        return item in self
