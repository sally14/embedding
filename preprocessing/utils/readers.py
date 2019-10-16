"""
Contains utils functions to parse args from docopt (not always in the right format)
"""
import os
import logging

logger = logging.getLogger('preprocessor')


def readArgs(args):
    """Reads and correct docopt parsed arguments from the command-line interface.
    For instance, int, floats, booleans need to be converted back to their
    type.

    Args
        args : dic
            docopt generated argument dictionnary
    Returns
        params : dic
            the dictionnary of corrected arguments
    """
    params = {}
    for k in args.keys():
        k2 = k.replace("<", "").replace(">", "").replace("-", "")
        try:  # Convert strings to int or floats when required
            params[k2] = int(args[k])
        except:
            try:
                params[k2] = float(args[k])
            except:
                try:
                    params[k2] = str2bool(args[k])
                except:
                    params[k2] = args[k]
    return params


def str2bool(s):
    """Helps to convert a string representing a boolean into the boolean"""
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError


def checkExistenceDir(path):
    """Checks if a given path is the path of a directory, and if not, creates
    the directory.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        logger.warning('Directory does not seem to exist, creating one.')
        os.mkdir(path)


def checkExistenceFile(path):
    """Checks if a given path is the path of a directory.
    """
    path = os.path.abspath(path)
    return os.path.isfile(path)
