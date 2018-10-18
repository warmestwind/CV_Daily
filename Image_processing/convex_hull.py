# https://startupnextdoor.com/computing-convex-hull-in-python/
from collections import namedtuple
import matplotlib.pyplot as plt
import random

Point = namedtuple('Point', 'x y')


class ConvexHull(object):
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is counter- clockwise of p2.
        目的是找逆时针方向的p2，即函数返回值>0的点

        # # https: // www.geeksforgeeks.org / check - if -two - given - line - segments - intersect /
        # # 斜率相减  p1p0 ,p2p0
        # # >0 :clockwise
        # # <0 :counterclockwise
        # # =0 : colinear
        '''
        difference = (
            (p1.y-origin.y)* (p2.x-origin.x)- (p2.y-origin.y)*(p1.x-origin.x)

        )


        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points #namedtuple('point',['x','y'])

        # get leftmost point
        start = points[0] #leftmost point
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start: #转了一圈

            # get the first point (initial max) to use to compare with others
            # 在点集里找一个非origin/start/pivot点的初始点p1
            p1 = None
            for p in points:
                if p is point: # is 比较两个对象是否相同（内存地址）， ==比较两个对象是否内容相同
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction < 0: # p2相对p1为逆时针
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        # https://blog.csdn.net/qiu931110/article/details/68130199
        plt.scatter(x, y, marker='D', c= 'g')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy, c= 'r')

        plt.title('Convex Hull')
        plt.show()


def main():
    ch = ConvexHull()
    for _ in range(10):
        ch.add(Point(random.randint(-100, 100), random.randint(-100, 100)))

    print("Points on hull:", ch.get_hull_points())
    ch.display()


if __name__ == '__main__':
    main()
