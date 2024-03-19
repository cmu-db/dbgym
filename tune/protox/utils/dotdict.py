class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
