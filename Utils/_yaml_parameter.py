import yaml

__all__ = [
    'yaml_params'
]

class ParameterContainer:
    def __repr__(self):
        ret = ''
        ret += '------------------ Parameters ------------------\n'
        for k, v in self.__dict__.items():
            if k != '__repr__':
                ret += '{}:\t{}\n'.format(k, v)
        ret += '------------------------------------------------'
        return ret

def dict2container(d):
    ret = ParameterContainer()
    for k, v in d.items():
        ret.__dict__[k] = dict2container(v)\
            if isinstance(v, dict)\
            else v
    return ret

def container2dict(c):
    ret = dict()
    for k, v in c.__dict__.items():
        ret[k] = container2dict(v)\
            if isinstance(v, ParameterContainer)\
            else v


def yaml_params(param_file):
    return dict2container(yaml.safe_load(open(param_file)))