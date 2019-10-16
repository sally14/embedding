"""
Contains utils functions to parse args from docopt (not always in the right format)
"""


def readArgs(argsDic):
    params = {}
    for k in argsDic.keys():
        k2 = k.replace("<", "").replace(">", "").replace("-", "")
        try:
            params[k2] = int(argsDic[k])
        except:
            try:
                params[k2] = float(argsDic[k])
            except:
                params[k2] = argsDic[k]
    return params
