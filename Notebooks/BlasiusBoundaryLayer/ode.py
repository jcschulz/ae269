import numpy as np

def euler(fun, y0, t_span):
    """
    Euler's method for solving a system of ordinary differential equations.
    The method  is first-order accurate with the global error proportional
    to the step-size.

    Args:
        fun (callable) : Right-hand side of the ODE system. The calling signature
            is `fun(t,y)`, where `t` is a scalar.
        y0 (array_like) : Initial state.
        t_span (array_like) : Interval of integration (t0, tf).
    """

    y = np.zeros((t_span.size, y0.size))
    y[0,:] = y0

    for i in range(len(t_span)-1):
        dt = t_span[i+1] - t_span[i]
        y[i+1,:] = y[i,:] + np.dot(dt, fun(t_span[i], y[i,:]))

    return y