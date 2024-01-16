def numerify(var):
    """
    Cast an object as an int or float if possible. Otherwise return unmodified
    """

    if isinstance(var, str):
        try:
            var = int(var)

        except:
            try:
                var = float(var)
            except:
                pass
    else:
        pass

    return var
