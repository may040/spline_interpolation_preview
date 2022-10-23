#!/usr/bin/python

from cagd.polyline import polyline
from cagd.spline import spline, get_point
from cagd.vec import vec2
import cagd.scene_2d as scene_2d

if __name__ == "__main__":
    # pts = [vec2(0, .4), vec2(.8, .8), vec2(.5, 1.2), vec2(-.03, .4), vec2(.4, 0), vec2(1, .2)]
    # spl = spline(3)
    # s1 = spl.interpolate_cubic(spline.INTERPOLATION_CHORDAL, pts)
    # s1.set_color("#0000ff")

    pts = [vec2(0, 2.5), vec2(-1, 1), vec2(1, -1), vec2(0, -2.5), vec2(-1, -1), vec2(1, 1)]
    example_spline = spline(3)
    s1 = example_spline.interpolate_cubic_periodic(pts)

    sc = scene_2d.scene()
    sc.set_resolution(900)
    sc.add_element(s1)

    for i in [1, -1]:
        para = s1.generate_parallel(i * 0.04, 0.001)
        para.set_color("#999999")
        sc.add_element(para)

    sc.write_image()
    sc.show()
