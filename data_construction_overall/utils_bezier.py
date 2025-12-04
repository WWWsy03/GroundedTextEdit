import numpy as np

def bezier(t, p0, p1, p2, p3):
    return (
        (1-t)**3*np.array(p0)
        + 3*(1-t)**2*t*np.array(p1)
        + 3*(1-t)*t**2*np.array(p2)
        + t**3*np.array(p3)
    )

def bezier_tangent(t, p0, p1, p2, p3):
    return (
        -3*(1-t)**2*np.array(p0)
        + (3*(1-t)**2 - 6*(1-t)*t)*np.array(p1)
        + (6*(1-t)*t - 3*t**2)*np.array(p2)
        + 3*t**2*np.array(p3)
    )

def get_text_width(text, font, tracking=0):
    char_widths = [font.getlength(c) for c in text]
    return sum(char_widths) + (len(text) - 1) * tracking, char_widths