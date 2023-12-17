import os
import geometry as geom
from combinatorics import *
from math import sin, cos, pi

def draw(obj):
    if type(obj) is ChordDiagram:
        ans = []
        ans.append(geom.draw_circle())
        for chord in obj.chords_coord:
            p = (cos(chord[0] * pi / (obj.order)), sin(chord[0] * pi / (obj.order)))
            q = (cos(chord[1] * pi / (obj.order)), sin(chord[1] * pi / (obj.order)))
            ans.append(geom.draw_chord(p, q))
        return '\n'.join(ans)
    # if type(obj) is StrandDiagram:
    #     ans = []
    #     ans.append(geom.draw_circle(n=obj.numstrands))
    if type(obj) is Graph:
        ans = []
        for edge in obj.edges:
            p = (cos(edge[0] * 2 * pi / (obj.order)), sin(edge[0] * 2 * pi / (obj.order)))
            q = (cos(edge[1] * 2 * pi / (obj.order)), sin(edge[1] * 2 * pi / (obj.order)))
            ans.append(geom.draw_line(p, q))
        for vertex in obj.vertices:
            p = (cos(vertex * 2 * pi / (obj.order)), sin(vertex * 2 * pi / (obj.order)))
            ans.append(geom.draw_point(p))
        return '\n'.join(ans)


def create_picture(obj, name='picture'):
    special_symbol = '% ---!!!---'
    with open('template_draw.tex', 'r') as f_in, open(f'{name}.tex', 'w') as f_out:
        for line in f_in:
            line = line.rstrip()
            if line[:len(special_symbol)] != special_symbol:
                print(line, file = f_out)
            else:
                print(draw(obj), file = f_out)
    os.system(f"pdflatex {name}.tex > output.txt")
    for res in ['tex', 'aux', 'log']:
        os.remove(f"{name}.{res}")
