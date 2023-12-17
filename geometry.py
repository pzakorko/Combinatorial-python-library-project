from typing import Union, List
from math import sqrt, atan2, pi, sin, cos

def inner_prod(p, q):
    return sum([p[i] * q[i] for i in range(len(p))])

def norm(p):
    return sqrt(inner_prod(p, p))

def draw_circle(center=(0,0), radius=1):
    return f"\\draw [thick] {str(center)} circle [radius={radius}];"

def draw_point(center, radius=0.05):
    return f"\\draw [fill] {str(center)} circle [radius={radius}];"

def draw_line(p, q):
    return f"\\draw {repr(p)} -- {repr(q)};"

def draw_chord(p: tuple, q: tuple, radius=1, center=(0,0)):
    pq2 = tuple((p[i] + q[i]) / 2 for i in range(2))
    if round(pq2[0] - center[0], 5) == 0 and round(pq2[1] - center[1], 5) == 0:
        return draw_line(p, q)
    else:
        c_new = tuple(pq2[i] * (radius / norm(pq2)) ** 2 for i in range(2))
        r_new = norm(tuple(c_new[i] - p[i] for i in range(2)))
        alpha_p = atan2(*tuple(p[i] - c_new[i] for i in range(1,-1,-1))) * 180 / pi
        alpha_q = atan2(*tuple(q[i] - c_new[i] for i in range(1,-1,-1))) * 180 / pi
        p = tuple(round(p[i], 3) for i in range(2))
        q = tuple(round(q[i], 3) for i in range(2))
        if alpha_p < 0:
            alpha_p += 360
        if alpha_q < 0:
            alpha_q += 360
        if alpha_p - alpha_q > 180 or (alpha_q - alpha_p > 0 and alpha_q - alpha_p < 180):
            if alpha_q < alpha_p:
                alpha_q += 360
            return f"\\draw {str(p)} arc ({str(round(alpha_p, 3))}:{str(round(alpha_q, 3))}:{str(round(r_new, 3))});"
        if alpha_p < alpha_q:
            alpha_p += 360
        return f"\\draw {str(q)} arc ({str(round(alpha_q, 3))}:{str(round(alpha_p, 3))}:{str(round(r_new, 3))});"

