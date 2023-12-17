from math import sqrt
from typing import List, Union
import itertools
import os
import geometry as geom
from math import sin, cos, pi

class Permutation():
    def __init__(self, *image):
        self.length = len(image)
        self.image = image
        if [i + 1 for i in range(self.length)] != sorted(self.image):
            raise Exception("Incorrect permutation")

    def len(self):
        return self.length

    def im(self):
        return self.image

    def add_length(self, add=1) -> "Permutation":
        if add == 0:
            return self
        return Permutation(*self.image, self.length + 1).add_length(add-1)

    def isId(self) -> bool:
        for i in range(self.length):
            if self.image[i] != i + 1:
                return False
        return True

    def inverse(self) -> "Permutation":
        ans = [0 for _ in range(self.length)]
        for i in range(self.length):
            ans[self.image[i] - 1] = i + 1
        return Permutation(*ans)

    def cyclic(self):
        i = 0
        if self.isId():
            return []
        cycle = []
        while True:
            if self.image[i] != i + 1:
                cycle.append(i + 1)

    @staticmethod
    def cycle(c, n):
        perm = [i + 1 for i in range(n)]
        if len(c) <= 1:
            return Permutation(*perm)
        for i in range(len(c)):
            perm[c[i - 1] - 1] = c[i]
        return Permutation(*perm)

    @staticmethod
    def all(n):
        if n == 1:
            return [Permutation(1)]
        ans = []
        for perm in Permutation.all(n - 1):
            for i in range(n, 0, -1):
                ans.append(Permutation.cycle((i, n), n) * perm.add_length())
        return ans

    @staticmethod
    def cyclic(*cycles, order=0):
        """Permutation given by a cyclic decomposition"""
        m = max(itertools.chain(*cycles))
        if order == 0:
            order = m
        if m > order:
            raise Exception("Incorrect permutation group was given")
        ans = [i + 1 for i in range(order)]
        error = [0 for i in range(m)]
        for cycle in cycles:
            for i in range(len(cycle)):
                if not error[cycle[i] - 1]:
                    error[cycle[i] - 1] = 1
                else:
                    raise Exception("Incorrect cyclic decomposition")
                ans[cycle[i - 1] - 1] = cycle[i]
        return Permutation(*ans)

    def __mul__(self, other: "Permutation") -> "Permutation":
        if self.length != other.length:
            raise Exception("Permutations are in different groups")
        return Permutation(*[self.image[other.image[i] - 1] for i in range(self.length)])

    def __str__(self):
        return repr(self.image)

    def __repr__(self):
        return f'Permutation{repr(self.image)}'

class Graph():
    def __init__(self, *edges, order = 0):
        new_edges = []
        for edge in edges:
            if len(edge) != 2:
                raise Exception("An edge should be given by two vertices")
            for i in range(2):
                if not type(edge[i]) is int or edge[i] <= 0:
                    raise Exception("Vertices should be labelled by positive integers")
            if edge[0] == edge[1]:
                raise Exception("No loops in a graph are allowed")
            new_edges.append((min(edge), max(edge)))
        if new_edges:
            m = max(itertools.chain(*new_edges))
        else:
            m = 0
        if order == 0:
            order = m
        self.oriented = False
        self.edges = sorted(new_edges)
        if order < m:
            raise Exception("Wrong order of the graph was given")
        self.vertices = list(range(1, order + 1))
        self.ord = order

    def isdiscrete(self):
        return not self.edges

    def delete_vertex(self, vertex):
        if vertex <= 0 or vertex > self.ord:
            raise Exception("No such vertex in a graph")
        ans = []
        for edge in self.edges:
            new_edge = []
            for i in range(2):
                if edge[i] == vertex:
                    new_edge = []
                    break
                if edge[i] > vertex:
                    new_edge.append(edge[i] - 1)
                if edge[i] < vertex:
                    new_edge.append(edge[i])
            if new_edge:
                ans.append(new_edge)
        return Graph(*ans, order = self.ord-1)

    def delete_edge(self, edge):
        ans = list(self.edges).copy()
        if edge in ans:
            del ans[ans.index(edge)]
        if not self.oriented and reversed(edge) in ans:
            del ans[ans.index(reversed(edge))]
        return Graph(*ans, order = self.ord)

    def add_vertices(self, n):
        if n < 0:
            raise Exception("Number of additional vertices should be positive")
        return Graph(*self.edges, order=self.ord + n)

    def add_edges(self, *lst_edges):
        if not lst_edges:
            m = self.ord
        else:
            m = max(max(itertools.chain(*lst_edges)), self.ord)
        return Graph(*list(set(self.edges) | set(lst_edges)), order = m)

    def neighbors(self, vertex):
        ans = []
        for edge in self.edges:
            for i in range(2):
                if edge[i] == vertex:
                    ans.append(edge[1-i])
        return ans

    def contract(self, edge):
        edge = (min(edge), max(edge))
        if edge not in self.edges:
            raise Exception("No such edge in a graph")
        new_edges = []
        for nghbr in self.neighbors(edge[1]):
            if nghbr > edge[1] and nghbr != edge[0]:
                new_edges.append((edge[0], nghbr - 1))
            if nghbr < edge[1] and nghbr != edge[0]:
                new_edges.append((edge[0], nghbr))
        return self.delete_vertex(edge[1]).add_edges(*new_edges)

    def __str__(self):
        return str(self.edges)

    def __mul__(self, other):
        list_edges = self.edges + [(edge[0] + self.ord, edge[1] + self.ord) for edge in other.edges]
        return Graph(*list_edges, order = self.ord + other.ord)

    def __repr__(self):
        return f"Graph({str(self.edges)[1:-1]}, order = {self.ord})"

# def StrandDiagram():
#     def __init__(self, chords):
#         self.numstrands = len(chords)
#         self.chords = chords
#         alpha = 90 - 90 / n
#         d_alpha = 180 / n
#         for i in range(n):
                    
#         self.points_coord = ans

class ArcDiagram():
    def __init__(self, *chords: int):
        self.ord = len(chords) // 2
        self.chords = chords

        if len(chords) % 2 == 1:
            raise Exception("Incorrect diagram: odd number of endpoints")
        for chord in range(1, len(chords) // 2 + 1):
            if chords.count(chord) != 2:
                raise Exception(f"Chord {i} has {chords.count(chord)} endpoints instead of 2")

        ans, boollist = [0 for _ in range(self.ord)], []
        for i in range(2 * self.ord):
            if self.chords[i] in boollist:
                ans[self.chords[i] - 1].append(i)
            else:
                boollist.append(self.chords[i])
                ans[self.chords[i] - 1] = [i]
        self.chords_coord = ans
        self.numline = ''.join(map(str, chords))

    def Chords(self):
        return self.chords_coord

    def endpoints(self):
        return self.chords

    def find(self, chord):
        self.correct_chord(chord)
        ans = []
        for point in range(2 * self.ord):
            if self.chords[point] == chord:
                ans.append(point)
        return ans

    def order(self):
        return self.ord

    def isempty(self) -> bool:
        return not bool(self.chords)

    def correct_chord(self, index):
        if index > self.ord:
            raise Exception("There are less chords in a diagram than you think")
        if index <= 0:
            raise Exception("All chords are labelled by positive integers")

    def correct_point(self, point):
        if point >= 2 * self.ord:
            raise Exception("There are less chords in a diagram than you think")
        if point < 0:
            raise Exception("All endpoints are labelled by non-negative integers")

    def chord_length(self, chord):
        return abs(self.Chords()[chord-1][0] - self.Chords()[chord-1][1]) - 1

    def another_end(self, pt):
        self.correct_point(pt)
        crd = self.chords[pt]
        for point, chord in enumerate(self.chords):
            if point != pt and chord == crd:
                return point

    def intersectbool(self, chord_i, chord_j):
        """
        True means chords i and j intersect
        False means there is no intersection
        """
        if chord_i == chord_j:
            return True
        self.correct_chord(chord_i)
        self.correct_chord(chord_j)
        endpoints_i, endpoints_j = self.find(chord_i), self.find(chord_j)
        if (endpoints_i[0] < endpoints_j[0] and endpoints_j[0] < endpoints_i[1] and endpoints_i[1] < endpoints_j[1]) or (endpoints_j[0] < endpoints_i[0] and endpoints_i[0] < endpoints_j[1] and endpoints_j[1] < endpoints_i[1]):
            return True
        return False

    def normal_form(self):
        ans, dct, chord_number = [], {}, 1
        for chord in self.chords:
            if chord in dct:
                ans.append(dct[chord])
            else:
                ans.append(chord_number)
                dct[chord] = chord_number
                chord_number += 1
        return ArcDiagram(*ans)

    def permutation_form(self) -> "Permutation":
        perm = []
        for point1 in range(2 * self.ord):
            for point2 in range(2 * self.ord):
                if point1 != point2 and self.chords[point1] == self.chords[point2]:
                    perm.append(point2 + 1)
                    break
        return Permutation(*perm)

    def shift(self, direction='right'):
        if self.isempty():
            return self
        if direction == 'right':
            return ChordDiagram(self.chords[-1], *self.chords[:-1])
        if direction == 'left':
            return ChordDiagram(*self.chords[1:], self.chords[0])
        raise Exception(f"'{direction}' is an incorrect direction to shift")

    def __eq__(self, other: "ArcDiagram"):
        return self.normal_form().chords == other.normal_form().chords

    def __mul__(self, other: "ArcDiagram"):
        return ArcDiagram(*(self.chords + tuple(other.chords[i] + self.ord for i in range(2 * other.ord))))

    def __str__(self):
        return repr(self.chords)

    def __repr__(self):
        return f'ArcDiagram{repr(self.chords)}'

class ChordDiagram(ArcDiagram):

    def delete(self, chord):
        self.correct_chord(chord)
        ans = []
        for point in range(2 * self.ord):
            if self.chords[point] > chord:
                ans.append(self.chords[point] - 1)
            if self.chords[point] < chord:
                ans.append(self.chords[point])
        return ChordDiagram(*ans)

    # def transpose(self, chord_i, chord_j):
    #     """
    #     Creates a new chord diagram with swapped endpoints of chords i and j
    #     """
    #     if i == j:
    #         return self
    #     self.correct_chord(chord_i)
    #     self.correct_chord(chord_j)
    #     endpoints_i, endpoints_j = self.find(chord_i), self.find(chord_j)
    #     ans = self.chords
    #     ans[endpoints_i[0]], ans[endpoints_j[0]] = chord_j, chord_i
    #     return ChordDiagram(*ans)

    def transpose(self, i, j):
        ans = [self.chords[k] for k in range(2 * self.ord)]
        ans[i], ans[j] = ans[j], ans[i]
        return ChordDiagram(*ans)

    def normal_form(self):
        ans, dct, chord_number = [], {}, 1
        for chord in self.chords:
            if chord in dct:
                ans.append(dct[chord])
            else:
                ans.append(chord_number)
                dct[chord] = chord_number
                chord_number += 1
        return ChordDiagram(*ans)

    def intersection_graph(self):
        ans = []
        for i in range(1, self.ord + 1):
            for j in range(i + 1, self.ord + 1):
                if self.intersectbool(i, j):
                    ans.append([i, j])
        return Graph(*ans)

    def __repr__(self):
        return f'ChordDiagram{repr(self.chords)}'

    def __mul__(self, other: "ArcDiagram"):
        return ChordDiagram(*(self.chords + tuple(other.chords[i] + self.ord for i in range(2 * other.ord))))

    def __eq__(self, other: "ChordDiagram"):
        if self.ord != other.ord:
            return False
        for _ in range(2 * self.ord):
            equal_bool = True
            for point in range(2 * self.ord):
                if self.another_end(point) != other.another_end(point):
                    equal_bool = False
                    break
            if equal_bool:
                return True
            other = other.shift()
        return False

class Polynomial():
    def __init__(self, *coeff, var='c'):
        if not coeff:
            self.coeff = [0]
        else:
            if isinstance(coeff[0], list) or isinstance(coeff[0], tuple):
                self.coeff = coeff[0]
            else:
                self.coeff = coeff
        self.var = var

    def __repr__(self):
        return repr(f'Polynomial(({", ".join(map(str, self.coeff))}))')

    def __str__(self):
        outside_poly_bool = True
        for i in range(len(self.coeff) - 1, -1, -1):
            if outside_poly_bool and self.coeff[i]:
                outside_poly_bool = False
                ans = ''
                if self.coeff[i] < 0:
                    ans += '-'
            elif self.coeff[i]:
                if self.coeff[i] < 0:
                    ans += ' - '
                else:
                    ans += ' + '
            if self.coeff[i]:
                if abs(self.coeff[i]) != 1:
                    if isinstance(self.coeff[i], int):
                        ans += str(abs(self.coeff[i]))
                    else:
                        ans += str(round(abs(self.coeff[i]), 3))
                if abs(self.coeff[i]) == 1 and i == 0:
                    ans += '1'
                if i == 1:
                    ans += self.var
                elif i:
                    ans += self.var + '^' + str(i)
        if outside_poly_bool:
            return '0'
        else:
            return ans

    def iszero(self):
        for c in self.coeff:
            if c:
                return False
        return True

    def degree(self):
        for i in range(len(self.coeff) - 1, -1, -1):
            if self.coeff[i] != 0:
                return i

    def addsub(self, other, param=1):
        ans = []
        for i in range(min(len(self.coeff), len(other.coeff))):
            ans.append(self.coeff[i] + param * other.coeff[i])
        if len(self.coeff) > len(other.coeff):
            ans += self.coeff[len(other.coeff):]
        elif len(self.coeff) < len(other.coeff):
            ans += [param * other.coeff[i] for i in range(len(self.coeff), len(other.coeff))]
        return Polynomial(ans)

    def multiplication(self, other):
        ans = [0 for _ in range(len(self.coeff) * len(other.coeff))]
        for i in range(len(self.coeff)):
            for j in range(len(other.coeff)):
                ans[i + j] += self.coeff[i] * other.coeff[j]
        return Polynomial(ans)

    def poly_from_str(string):
        lst = string.split()
        ans = []
        for el in lst:
            if el.isnumeric() or (el[0]=='-' and el[1:].isnumeric()):
                ans.append(int(el))
            else:
                ans.append(float(el))
        return Polynomial(ans)

    def coeff(self, i):
        return self.coeff[i]

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self + Polynomial(other)
        return self.addsub(other)

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Polynomial(other) + self
        return self.addsub(other)

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self - Polynomial(other)
        return self.addsub(other, param =-1)

    def __rsub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Polynomial(other) - self
        return other.addsub(self, param = -1)

    def __eq__(self, other):
        return (self - other).iszero()

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * Polynomial(other)
        return self.multiplication(other)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * Polynomial(other)
        return self.multiplication(other)


def draw(obj):
    if type(obj) is ChordDiagram:
        ans = []
        ans.append(geom.draw_circle())
        for chord in obj.chords_coord:
            p = (cos(chord[0] * pi / (obj.ord)), sin(chord[0] * pi / (obj.ord)))
            q = (cos(chord[1] * pi / (obj.ord)), sin(chord[1] * pi / (obj.ord)))
            ans.append(geom.draw_chord(p, q))
        return '\n'.join(ans)
    if type(obj) is Graph:
        ans = []
        for edge in obj.edges:
            p = (cos(edge[0] * 2 * pi / (obj.ord)), sin(edge[0] * 2 * pi / (obj.ord)))
            q = (cos(edge[1] * 2 * pi / (obj.ord)), sin(edge[1] * 2 * pi / (obj.ord)))
            ans.append(geom.draw_line(p, q))
        for vertex in obj.vertices:
            p = (cos(vertex * 2 * pi / (obj.ord)), sin(vertex * 2 * pi / (obj.ord)))
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

def chromatic(graph: Graph):
    if type(graph) != Graph:
        raise TypeError("A given object is not a graph")
    if graph.isdiscrete():
        ans = [0 for i in range(graph.ord + 1)]
        ans[-1] = 1
        return Polynomial(ans, var='c')
    edge = graph.edges[0]
    return chromatic(graph.delete_edge(edge)) - chromatic(graph.contract(edge))

def wsl2(diagram):
    if diagram.intersection_graph().isdiscrete():
        ans = [0 for i in range(diagram.ord + 1)]
        ans[-1] = 1
        return Polynomial(ans, var='c')
    min_length = 2 * diagram.order()
    for chord in range(1, diagram.order() + 1):
        if diagram.chord_length(chord) < min_length:
            min_length = diagram.chord_length(chord)
            min_chord = chord
    if min_length == 0:
        return Polynomial(0, 1) * wsl2(diagram.delete(min_chord))
    elif min_length == 1:
        return Polynomial(-1, 1) * wsl2(diagram.delete(min_chord))
    else:
        i, j = diagram.Chords()[min_chord - 1]
        i1, j1 = i + 1, j - 1
        i2, j2 = diagram.another_end(i1), diagram.another_end(j1)
        chord_i, chord_j = diagram.endpoints()[i1], diagram.endpoints()[j1]
        D1 = diagram.transpose(i, i1)
        D2 = diagram.transpose(j, j1)
        D3 = D2.transpose(i, i1)
        D4 = diagram.transpose(i1, j2).delete(min_chord)
        D5 = diagram.transpose(i1, j1).delete(min_chord)
        return wsl2(D1) + wsl2(D2) - wsl2(D3) + wsl2(D4) - wsl2(D5)


