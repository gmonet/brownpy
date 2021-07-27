import math

def prefix(x, dimension=1):
    """Give the number an appropriate SI prefix.

    :param x: Too big or too small number.
    :returns: String containing a number between 1 and 1000 and SI prefix.
    https://stackoverflow.com/questions/29627796/pretty-printing-physical-quantities-with-automatic-scaling-of-si-prefixes
    """
    if x == 0:
        return "0  "

    l = math.floor(math.log10(abs(x)))

    div, mod = divmod(l, 3*dimension)
    return "%.3g %s" % (x * 10**(-l + mod), " kMGTPEZYyzafpnµm"[div])