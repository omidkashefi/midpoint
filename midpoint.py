from math import sin, cos, sqrt, atan2, radians, pi
import pandas as pd
import numpy as np
import optunity
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

def function_to_optimize(lat, lon):
    d = 0
    midpoint = (lat, lon)
    for p, w in zip(points, weights):
        d += w * distance(p, midpoint)

    return d

def optimize(startin_point):
    print "\nBegin optimization"
    midpoint = startin_point
    constraints = {'lat':[midpoint[0]-0.1 , midpoint[0]+0.1], 'lon': [midpoint[1]-0.1 , midpoint[1]+0.1]}

    print "\tStarting from:\t\t", midpoint
    print "\tCurrent distance:\t", function_to_optimize(midpoint[0], midpoint[1])[0]

    for sname in optunity.available_solvers(): #['particle swarm']
        #create a solver
        suggestion = optunity.suggest_solver(num_evals=500, solver_name=sname, **constraints)
        solver = optunity.make_solver(**suggestion)

        #optimize the function
        optimum = optunity.optimize(solver, function_to_optimize, maximize=False, max_evals=100)

        print "\n\t==================================="
        print "\tSolver name:\t", suggestion['solver_name']
        print "\tMidpoint:\t", [optimum[0]['lat'], optimum[0]['lon']]
        print "\tDistance:\t", optimum[1][0][0]
        print "\tIterations:\t", optimum[1][1]['num_evals']
        #print "\tTime (ms):\t", optimum[1][1]['time']

        #print optimum

points = []
weights = []
def main(**args):
    global points, weights

    points, weights = readData(args['db'])
    print "\nLoading {} points\n".format(len(points))

    midpoint1 = points.mean(axis=0)
    print "Midpoint (average latitude/longitude):\t", midpoint1

    c_points = convertoToCartesian(points)
    c_midpoint = c_points.mean(axis=0)
    midpoint2 = convertToDegree([c_midpoint])[0]
    print "Midpoint (center of gravity):\t\t", midpoint2

    c_midpoint = weightedMean(c_points, weights)
    midpoint3 = convertToDegree([c_midpoint])[0]
    print "Weighted midpoint (center of gravity):\t", midpoint3

    optimize(midpoint3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute midpoint of set of points', version='%(prog)s 1.0')
    parser.add_argument('--db', type=str, default='data.csv', required=True, help='Path to the dataset in csv format [., latitude, longitude, weight, .*]')
    args = parser.parse_args()

    main(**vars(args))
