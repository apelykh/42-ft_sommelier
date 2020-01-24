def ft_dot(a, b):
    """
    Dot product of two sequences of numbers. Both arguments have to
    be of equal length.
    
    :param np.array a:
    :param np.array b:
    """

    if len(a) != len(b):
        raise ValueError('arguments have different length')

    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    
    return result
