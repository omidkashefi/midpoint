from math import sin, cos, sqrt, atan2, radians, pi
import pandas as pd
import numpy as np
import argparse


def readData(filename):
    points = pd.read_csv(filename, delimiter=",", header=0, usecols=[1,2]).as_matrix()
    weights = pd.read_csv(filename, delimiter=",", header=0, usecols=[3]).as_matrix()
    return points, weights

def distance(p1, p2):
    R = 6373.0

    lat1 = radians(p1[0])
    lon1 = radians(p1[1])
    lat2 = radians(p2[0])
    lon2 = radians(p2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def convertoToCartesian(points):
    cartesian_points = []
    for p in points:
        lat = radians(p[0])
        lon = radians(p[1])

        x = cos(lat) * cos(lon)
        y = cos(lat) * sin(lon)
        z = sin(lat)

        cartesian_points.append([x, y, z])

    return np.array(cartesian_points)

def convertToDegree(points):
    degree_points = []
    for p in points:
        x = p[0]
        y = p[1]
        z = p[2]
        lon = atan2(y, x)
        hyp = sqrt(x*x + y*y)
        lat = atan2(z, hyp)

        lat *= 180 / pi
        lon *=  180 / pi

        degree_points.append([lat, lon])

    return np.array(degree_points)

def weightedMean(points, weights):
    X = 0.0
    Y = 0.0
    Z = 0.0
    W = 0.0
    for p, w in zip(points, weights):
        x = p[0]
        y = p[1]
        z = p[2]

        X += x * w
        Y += y * w
        Z += z * w
        W += w

    return [X/W, Y/W, Z/W]


def main(**args):
    points, weights = readData(args['db'])
    print "Loading {} points".format(len(points))

    midpoint = points.mean(axis=0)
    print "Midpoint (average latitude/longitude):\t", midpoint

    c_points = convertoToCartesian(points)
    c_midpoint = c_points.mean(axis=0)
    midpoint = convertToDegree([c_midpoint])[0]
    print "Midpoint (center of gravity):\t\t", midpoint

    c_midpoint = weightedMean(c_points, weights)
    midpoint = convertToDegree([c_midpoint])[0]
    print "Weighted midpoint (center of gravity):\t", midpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute midpoint of set of points', version='%(prog)s 1.0')
    parser.add_argument('--db', type=str, default='data.csv', required=True, help='Path to the dataset in csv format [., latitude, longitude, weight, .*]')
    args = parser.parse_args()

    main(**vars(args))
