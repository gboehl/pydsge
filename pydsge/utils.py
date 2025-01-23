from sympy import Equality

def _dumper(obj, name=None, eq:str="=", indent:int=4, currindent:int=0):
    """Recursively dump (print) an object in a somewhat structured but human-readable manner.

    This function knows how to handle dictionaries, lists and tuples, as well as classes
    derived from them. For everything else, the usual stringification is used.

    Parameters
    ----------
    obj
        The object to be dumped.
    name (default: None)
        Name of the object to be used in output, if any.
    eq : str (default: "=")
        Equal sign to use for current (!) level
    indent : int (default: 4)
        Amount of whitespace to use for indenting when dumping recursively.
    currindent : int (default: 0)
        Amount of whitespace to start with when dumping.

    """

    if name is None:
        name_and_eq = ""
    else:
        name_and_eq = name + " " + eq + " "

    if isinstance(obj, dict):
        print(currindent*" " + name_and_eq + "{")
        for k, v in obj.items():
            _dumper(v, name=k, eq=":", indent=indent, currindent=currindent+indent)
        print(currindent*" " + "}")
    elif isinstance(obj, list):
        print(currindent*" " + name_and_eq + "[")
        for v in obj:
            _dumper(v, name=None, indent=indent, currindent=currindent+indent)
        print(currindent*" " + "]")
    elif isinstance(obj, tuple):
        print(currindent*" " + name_and_eq + "(")
        for v in obj:
            _dumper(v, name=None, indent=indent, currindent=currindent+indent)
        print(currindent*" " + ")")
    elif isinstance(obj, set):
        print(currindent*" " + name_and_eq + "{")
        for v in obj:
            _dumper(v, name=None, indent=indent, currindent=currindent+indent)
        print(currindent*" " + "}")
    elif isinstance(obj, Equality):
        print(currindent*" " + name_and_eq + str(obj.lhs) + " = " + str(obj.rhs))
    elif isinstance(obj, str):
        print(currindent*" " + name_and_eq + "'" + obj + "'")
    else:
        print(currindent*" " + name_and_eq + str(obj))

