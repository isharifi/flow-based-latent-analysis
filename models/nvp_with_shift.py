import types
from functools import wraps


def add_forward_with_shift(generator):
    def nvp_shifted(self, z, shift, *args, **kwargs):
        return self.forward(z=z + shift, *args, **kwargs)

    generator.nvp_shifted = types.MethodType(nvp_shifted, generator)


def nvp_with_shift(nvp_factory):
    @wraps(nvp_factory)
    def wrapper(*args, **kwargs):
        nvp = nvp_factory(*args, **kwargs)
        add_forward_with_shift(nvp)
        return nvp

    return wrapper
