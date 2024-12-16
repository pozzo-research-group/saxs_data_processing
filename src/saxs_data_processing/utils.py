import uuid


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


def is_valid_uuid(uuid_to_test, version=4):
    """
    Check if a test UUID is valid for the specified version
    """
    try:
        # check for validity of Uuid
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return True
