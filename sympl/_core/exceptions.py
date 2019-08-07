class InvalidStateError(Exception):
    pass


class InvalidPropertyDictError(Exception):
    pass


class SharedKeyError(Exception):
    pass


class DependencyError(Exception):
    pass


class ComponentMissingOutputError(Exception):
    pass


class ComponentExtraOutputError(Exception):
    pass
