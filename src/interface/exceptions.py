class InterfaceException(Exception):
    pass

class InterfaceOpenAIException(InterfaceException):
    pass

class UnknownModelException(InterfaceOpenAIException):
    pass