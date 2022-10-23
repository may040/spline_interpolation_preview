#! /usr/bin/python
import math

import numpy as numpy

import cagd.scene_2d as scene

from cagd.vec import vec2, vec3
from cagd.polyline import polyline
from cagd.bezier import bezier_surface, bezier_patches
import cagd.utils as utils
import copy
from math import *


def norm(v: vec2):
    return sqrt(pow(v.x, 2) + pow(v.y, 2))


def angle(p1: vec2, p2: vec2):
    dot_p = p1.dot(p2)

    return numpy.arccos(dot_p / (norm(p1) * norm(p2)))


def theta_hat(i: int, points: [vec2]):
    return min(pi - angle(points[i - 1], points[i]), pi / 2)  # todo: what is points[-1] ?


def d(i: int, points: [vec2]):
    if i == -1 or i == len(points) - 1: return 0
    return norm(points[i + 1] + points[i])


class spline:
    # Interpolation modes
    INTERPOLATION_EQUIDISTANT = 0
    INTERPOLATION_CHORDAL = 1
    INTERPOLATION_CENTRIPETAL = 2
    INTERPOLATION_FOLEY = 3

    def __init__(self, degree):
        assert (degree >= 1)
        self.degree = degree
        self.knots = None
        self.control_points = []
        self.color = "black"

    # checks if the number of knots, controlpoints and degree define a valid spline
    def validate(self):
        knots = self.knots.validate()
        points = len(self.knots) == len(self.control_points) + self.degree + 1
        return knots and points

    def evaluate(self, t):
        a, b = self.support()
        assert (a <= t <= b)
        if t == self.knots[len(self.knots) - self.degree - 1]:
            # the spline is only defined on the interval [a, b)
            # it is useful to define self(b) as lim t->b self(t)
            t = t - 0.000001
        return self.de_boor(t, 1)[0]

    # returns the interval [a, b) on which the spline is supported
    def support(self):
        return self.knots[self.degree], self.knots[len(self.knots) - self.degree - 1]

    def __call__(self, t):
        return self.evaluate(t)

    def tangent(self, t):
        a, b = self.support()
        assert (a <= t <= b)
        if t == self.knots[len(self.knots) - self.degree - 1]:
            # the spline is only defined on the interval [a, b)
            # it is useful to define self(b) as lim t->b self(t)
            t = t - 0.000001

        cps = self.de_boor(t, 2)
        diff = cps[1].__sub__(cps[0])
        diff_len = diff.__abs__()
        diff = vec2(diff.x / diff_len, diff.y / diff_len)
        return diff

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    # calculates the de_boor scheme at a given value t
    # stops when the column is only "stop" elements long
    # returns that column as a list
    def de_boor(self, t, stop):
        knot_i = self.knots.knot_index(t)
        deg = self.degree
        i = knot_i - deg

        # generate d_i^0
        d = [self.control_points[j + knot_i - deg] for j in range(0, deg + 1)]

        # loop through every column
        for k in range(1, deg + 1):
            # loop through every row
            for j in range(i, i + deg - k + 1):
                # calculate alpha_i^k
                alpha = (t - self.knots[j + k]) / (
                        self.knots[j + deg + 1] - self.knots[j + k])
                # calculate each element in the column
                d[j - i] = (1.0 - alpha) * d[j - i] + alpha * d[j + 1 - i]
            # if the column length equals stop, return the most recently calculated column
            if len(range(0, deg - k + 1)) == stop:
                return d[0: deg - k + 1]
        return None

    # adjusts the control points such that it represents the same function,
    # but with an added knot
    def insert_knot(self, t):
        knot_i = self.knots.knot_index(t) - self.degree
        new_cps = self.de_boor(t, 3)

        # delete the two old control points
        for i in range(self.degree - 1):
            self.control_points.pop(knot_i + 1)

        # adds the three new control points
        for i in range(self.degree):
            self.control_points.insert(knot_i + 1 + i, new_cps[i])

        self.knots.insert(t)

    def vector_strings(self, vectors: [vec2]):
        return str([(round(p.x, 2), round(p.y, 2)) for p in vectors])

    def get_axis_aligned_bounding_box(self):
        min_vec = copy.copy(self.control_points[0])
        max_vec = copy.copy(self.control_points[0])
        for p in self.control_points:
            # print("comparing {0} to {1} and {2}".format(p, min_vec, max_vec))
            if p.x < min_vec.x:
                min_vec.x = p.x
            if p.y < min_vec.y:
                min_vec.y = p.y
            if p.x > max_vec.x:
                max_vec.x = p.x
            if p.y > max_vec.y:
                max_vec.y = p.y
        return min_vec, max_vec

    def draw(self, scene, num_samples):
        i = self.degree - 1
        while i < len(self.knots) - self.degree - 2:
            i += 1
            k0 = self.knots[i]
            k1 = self.knots[i + 1]
            if k0 == k1:
                continue
            p0 = self(k0)
            for j in range(1, num_samples + 1):
                t = k0 + j / num_samples * (k1 - k0)
                p1 = self(t)
                scene.draw_line(p0, p1, self.color)
                p0 = p1

    def get_polyline_from_control_points(self):
        pl = polyline()
        for p in self.control_points:
            pl.append_point(p)
        return pl

    # generates a spline that interpolates the given points using the given mode
    # returns that spline object
    def interpolate_cubic(self, mode, points: [vec2]):
        ts = [float(x) for x in range(len(points))]
        # for each mode, generate the ^t array
        match mode:
            case self.INTERPOLATION_EQUIDISTANT:
                ts = [x for x in range(len(points))]
                pass
            case self.INTERPOLATION_CHORDAL:
                ts[0] = 0  # todo
                for i in range(0, len(points) - 1):
                    ts[i + 1] = norm(points[i + 1] - points[i]) + ts[i]
                pass
            case self.INTERPOLATION_CENTRIPETAL:
                ts[0] = 0  # todo
                for i in range(0, len(points) - 1):
                    ts[i + 1] = sqrt(norm(points[i + 1] - points[i])) + ts[i]
                pass
            case self.INTERPOLATION_FOLEY:
                ts[0] = 0  # todo
                for i in range(0, len(points) - 1):
                    summand_1 = (3 * theta_hat(i, points) * d(i - 1, points)) / (2 * d(i - 1, points) + d(i, points))
                    summand_2 = (3 * theta_hat(i + 1, points) * d(i + 1, points)) / (
                            2 * d(i + 1, points) + d(i, points))
                    ts[i + 1] = d(i, points) * (1 + summand_1 + summand_2) + ts[i]
                pass

        t = [ts[0], ts[0], ts[0]]
        t.extend(ts)
        t.extend([ts[-1], ts[-1], ts[-1]])

        m = len(points) - 1

        alpha = [(t[i + 2] - t[i]) / (t[i + 3] - t[i]) for i in range(2, m + 1)]
        beta = [(t[i + 2] - t[i + 1]) / (t[i + 3] - t[i + 1]) for i in range(2, m + 1)]
        gamma = [(t[i + 2] - t[i + 1]) / (t[i + 4] - t[i + 1]) for i in range(2, m + 2)]

        diag1 = [0.0 for _ in range(m + 3)]
        diag2 = [0.0 for _ in range(m + 3)]
        diag3 = [0.0 for _ in range(m + 3)]

        # set the first two lines of the tridiagonal matrix
        diag1[1] = -1
        diag2[0] = 1
        diag2[1] = 1 + alpha[0]
        diag3[0] = 0
        diag3[1] = -1 * alpha[0]

        # set the last two lines of the tridiagonal matrix
        diag1[-1] = 0
        diag1[-2] = -1 + gamma[-1]
        diag2[-1] = 1
        diag2[-2] = -1 * gamma[-1] + 2
        diag3[-2] = -1

        # set all other lines of the tridiagonal matrix
        for i in range(2, m + 1):
            diag1[i] = (1 - beta[i - 2]) * (1 - alpha[i - 2])
            diag2[i] = alpha[i - 2] * (1 - beta[i - 2]) + beta[i - 2] * (1 - gamma[i - 2])
            diag3[i] = beta[i - 2] * gamma[i - 2]

        res = [points[0], vec2(0, 0)]
        res.extend(points[1:-1])
        res.append(vec2(0, 0))
        res.append(points[-1])

        spl = spline(3)

        x = utils.solve_tridiagonal_equation(diag1, diag2, diag3, [p.x for p in res])
        y = utils.solve_tridiagonal_equation(diag1, diag2, diag3, [p.y for p in res])
        spl.control_points = [vec2(xp, yp) for (xp, yp) in zip(x, y)]

        spl.knots = knots(len(t))
        spl.knots.knots = t

        # print("t is {0} and controlpoints are {1}".format(t, [z.x for z in spl.control_points]))

        return spl

    # generates a spline that interpolates the given points and fulfills the definition
    # of a periodic spline
    # returns that spline object
    def interpolate_cubic_periodic(self, points):
        m = len(points)

        a = [1 / 6 for _ in points]
        b = [4 / 6 for _ in points]
        c = [1 / 6 for _ in points]

        x = utils.solve_almost_tridiagonal_equation(a, b, c, [p.x for p in points])
        y = utils.solve_almost_tridiagonal_equation(a, b, c, [p.y for p in points])

        n = 3
        spl = spline(n)
        cp = [vec2(xp, yp) for (xp, yp) in zip(x, y)]
        cp.extend([cp[0], cp[1], cp[2]])

        spl.control_points = cp
        t = [i for i in range(n + m + 4)]

        spl.knots = knots(len(t))
        spl.knots.knots = t
        return spl

    def get_parallel(self, dist):
        para_spline = spline(3)
        para_spline.knots = knots(len(self.knots.knots))
        para_spline.knots.knots = self.knots.knots

        kns = [x for x in self.knots.knots[3:-3]]

        tangents = [self.tangent(t) for t in kns]
        d = [[0, -1], [1, 0]]
        normals = [numpy.dot(d, [tangent.x, tangent.y]) for tangent in tangents]
        normals_dist = [dist * norms for norms in normals]

        points_old = [self.evaluate(t) for t in kns]
        points_new = [vec2(p.x + normal[0], p.y + normal[1]) for (normal, p) in zip(normals_dist, points_old)]

        return para_spline.interpolate_cubic(self.INTERPOLATION_CHORDAL, points_new)

    # for splines of degree 3, generate a parallel spline with distance dist
    # the returned spline is off from the exact parallel by at most eps
    def generate_parallel(self, dist, eps):
        copy = spline(3)
        copy.knots = knots(len(self.knots.knots))
        copy.knots.knots = [t for t in self.knots.knots]
        copy.control_points = [p for p in self.control_points]

        assert (copy.degree == 3)
        count = 1
        while count != 0:
            count = 0
            para = copy.get_parallel(dist)

            kns_self = copy.knots.knots[3:-3]
            kns_para = para.knots.knots[3:-3]

            for i in range(len(kns_self) - 1):
                t_self = kns_self[i] + (kns_self[i + 1] - kns_self[i]) / 2
                t_para = kns_para[i] + (kns_para[i + 1] - kns_para[i]) / 2

                real_distance = (copy.evaluate(t_self).__sub__(para.evaluate(t_para))).__abs__()
                if dist < 0:
                    real_distance = -real_distance

                # print("real distance: " + str(real_distance) + " dist: " + str(dist))
                if not (dist + eps > real_distance > dist - eps):
                    copy.insert_knot(t_self)
                    count += 1
                    break
        return para

    # generates a rotational surface by rotating the spline around the z axis
    # the spline is assumed to be on the xz-plane
    # num_samples refers to the number of interpolation points in the rotational direction
    # returns a spline surface object in three dimensions
    def generate_rotation_surface(self, num_samples):
        cps = [vec3(cp.x, 0, cp.y) for cp in self.control_points]

        b = [[vec3(0, 0, 0) for _ in range(num_samples)] for __ in range(len(cps))]

        for i in range(len(cps)):
            b[i] = self.get_circle_points(cps[i], num_samples)

        splines = [spline(3) for _ in range(len(cps))]

        for i in range(len(cps)):
            spl = spline(3)
            splines[i] = spl.interpolate_cubic_periodic(b[i])
            b[i] = [vec3(cp.x, cp.y, cps[i].z) for cp in splines[i].control_points]

        spl_sur = spline_surface((3, 3))
        spl_sur.knots = (self.knots, splines[0].knots)
        # = ([0,0,0,0,1.123,2.234,5.065, 5.065, 5.065, 5.065],[0,1,2,3,4,5,6,7,8,9,10,11,12])

        spl_sur.control_points = b
        return spl_sur

    def get_circle_points(self, point: vec3, num_samples):
        c = [vec3(0, 0, 0) for _ in range(num_samples)]
        for j in range(num_samples):
            c[j] = vec3(point.x * cos((2 * math.pi * j) / num_samples), point.x * sin((2 * math.pi * j) / num_samples),
                        point.z)
        return c


class spline_surface:
    # the two directions of the parameter space
    DIR_U = 0
    DIR_V = 1

    # creates a spline of degrees n,m
    # degree is a tuple (n,m)
    def __init__(self, degree):
        du, dv = degree
        assert (du >= 1 and dv >= 1)
        self.degree = degree
        self.knots = (None, None)  # tuple of both knot vectors
        self.control_points = [[]]  # 2dim array of control points

    # checks if the number of knots, controlpoints and degree define a valid spline
    def validate(self):
        if len(self.control_points) == 0:
            return False
        k1, k2 = self.knots
        d1, d2 = self.degree
        knots = k1.validate() and k2.validate()
        p1 = len(self.control_points)
        p2 = len(self.control_points[0])
        points1 = len(k1) == p1 + d1 + 1
        points2 = len(k2) == p2 + d2 + 1
        return knots and points1 and points2

    def evaluate(self, u, v):
        s1, s2 = self.support()
        a, b = s1
        c, d = s2
        assert (a <= u <= b and c <= v <= d)
        if u == b:
            u = u - 0.000001
        if v == d:
            v = v - 0.000001
        t = (u, v)
        return self.de_boor(t, (1, 1))[0][0]

    # return nested tuple ((a,b), (c,d))
    # the spline is supported in (u,v) \in [a,b)x[c,d]
    def support(self):
        k1, k2 = self.knots
        d1, d2 = self.degree
        s1 = (k1[d1], k1[len(k1) - d1 - 1])
        s2 = (k2[d2], k2[len(k2) - d2 - 1])
        return (s1, s2)

    def __call__(self, u, v):
        return self.evaluate(u, v)

    # calculates the de boor scheme at t = (u,v)
    # until there are only stop = (s1, s2) elements left
    def de_boor(self, t, stop):
        d1, d2 = self.degree
        k1, k2 = self.knots
        s1, s2 = stop
        u, v = t
        m1 = len(self.control_points)
        m2 = len(self.control_points[0])

        new_rows = [None for i in range(m1)]
        for row in range(m1):
            spl = spline(d2)
            spl.knots = k2
            spl.control_points = self.control_points[row]
            new_rows[row] = spl.de_boor(v, s2)

        new_pts = [None for i in range(s2)]
        for col in range(s2):
            spl = spline(d1)
            spl.knots = k1
            ctrl_pts = [new_rows[i][col] for i in range(m1)]
            spl.control_points = ctrl_pts
            new_pts[col] = spl.de_boor(u, s1)

        return new_pts

    def insert_knot(self, direction, t):
        if direction == self.DIR_U:
            self.__insert_knot_u(t)
        elif direction == self.DIR_V:
            self.__insert_knot_v(t)
        else:
            assert False

    def __insert_knot_v(self, t):
        du, dv = self.degree
        ku, kv = self.knots
        nu = len(self.control_points)
        nv = len(self.control_points[0])
        for i in range(nu):
            row = self.control_points[i]
            spl = spline(du)
            spl.control_points = copy.copy(row)
            spl.knots = copy.deepcopy(kv)
            spl.insert_knot(t)
            self.control_points[i] = spl.control_points
        kv.insert(t)

    def __insert_knot_u(self, t):
        du, dv = self.degree
        ku, kv = self.knots
        nu = len(self.control_points)
        nv = len(self.control_points[0])
        new_control_points = [[None for i in range(nv)] for j in range(nu + 1)]
        for i in range(nv):
            col = [self.control_points[j][i] for j in range(nu)]
            spl = spline(dv)
            spl.control_points = col
            spl.knots = copy.deepcopy(ku)
            spl.insert_knot(t)
            for j in range(nu + 1):
                new_control_points[j][i] = spl.control_points[j]
        self.control_points = new_control_points
        ku.insert(t)

    def to_bezier_patches(self):
        patches = bezier_patches()

        kns_u = [_ for _ in self.knots[0].knots]
        for t in kns_u[4:-4]:
            self.insert_knot(self.DIR_U, t)
            self.insert_knot(self.DIR_U, t)

        kns_v = [_ for _ in self.knots[1].knots]
        for t in kns_v[4:-4]:
            self.insert_knot(self.DIR_V, t)
            self.insert_knot(self.DIR_V, t)

        # cps = [cp[1:-1] for cp in self.control_points]
        cps = self.control_points
        for u in range(0, len(cps) - 1, 3):
            cps_u = cps[u]
            for v in range(0, len(cps_u) - 1, 3):
                patch_cps = [[cps[u][v+0], cps[u+1][v+0], cps[u+2][v+0], cps[u+3][v+0]],
                             [cps[u][v+1], cps[u+1][v+1], cps[u+2][v+1], cps[u+3][v+1]],
                             [cps[u][v+2], cps[u+1][v+2], cps[u+2][v+2], cps[u+3][v+2]],
                             [cps[u][v+3], cps[u+1][v+3], cps[u+2][v+3], cps[u+3][v+3]]
                             ]
                patch = bezier_surface((3,3))
                patch.control_points = patch_cps
                patches.append(patch)
        return patches


class knots:
    # creates a knots array with n elements
    def __init__(self, n):
        self.knots = [None for i in range(n)]

    def validate(self):
        prev = None
        for k in self.knots:
            if k is None:
                return False
            if prev is None:
                prev = k
            else:
                if k < prev:
                    return False
        return True

    def __len__(self):
        return len(self.knots)

    def __getitem__(self, i):
        return self.knots[i]

    def __setitem__(self, i, v):
        self.knots[i] = v

    def __delitem__(self, i):
        del self.knots[i]

    def __iter__(self):
        return iter(self.knots)

    def insert(self, t):
        i = 0
        while self[i] < t:
            i += 1
        self.knots.insert(i, t)

    def knot_index(self, v):
        # loop through all knots
        for i in range(len(self.knots)):
            # if v is between two knots at i and i + 1, return i
            if self.knots[i] <= v < self.knots[i + 1]:
                return i
        return None


def get_point(point: vec2, color="black", size=0.005):
    pl = polyline()
    pl.append_point(point.__add__(vec2(-size, -size)))
    pl.append_point(point.__add__(vec2(size, -size)))
    pl.append_point(point.__add__(vec2(size, size)))
    pl.append_point(point.__add__(vec2(-size, size)))
    pl.append_point(point.__add__(vec2(-size, -size)))
    pl.set_color(color)
    return pl


if __name__ == "__main__":
    points = [vec2(0, 0), vec2(1, 1), vec2(2, 2), vec2(0, 3)]
    spli = spline(3)
    spli = spli.interpolate_cubic(spline.INTERPOLATION_CHORDAL, points)
    spli_3d = spli.generate_rotation_surface(6)

    print(spli_3d.to_bezier_patches().export_off())

    # for cps in spli_3d.control_points:
    #     for i in range(len(cps)):
    #         cp = cps[i]
    #         print('\t' + str((round(cp.x, 2), round(cp.y, 2), round(cp.z, 2))) + ',')
