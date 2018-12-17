import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import csv
from math import sin, asin, cos, sqrt, atan2, radians, degrees
import matplotlib.pyplot as plt

from planning_utils import aStar, heuristic, createGrid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5
    PLANNING = 6


class MotionPlanning(Drone):

    def __init__(self, connection, destAddr):
        super().__init__(connection)

        self.targetPosition = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.inMission = True
        self.checkState = {}

        # initial state
        self.flightState = States.MANUAL
        self.destAddr = destAddr

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.localPositionCallback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocityCallback)
        self.register_callback(MsgID.STATE, self.stateCallback)

    def localPositionCallback(self):
        if self.flightState == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.targetPosition[2]:
                self.waypointTransition()
        elif self.flightState == States.WAYPOINT:
            if np.linalg.norm(self.targetPosition[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypointTransition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landingTransition()

    def velocityCallback(self):
        if self.flightState == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarmingTransition()

    def stateCallback(self):
        if self.inMission:
            if self.flightState == States.MANUAL:
                self.armingTransition()
            elif self.flightState == States.ARMING:
                if self.armed:
                    self.planPath()
            elif self.flightState == States.PLANNING:
                self.takeoffTransition()
            elif self.flightState == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manualTransition()

    def armingTransition(self):
        self.flightState = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoffTransition(self):
        self.flightState = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.targetPosition[2])

    def waypointTransition(self):
        self.flightState = States.WAYPOINT
        print("waypoint transition")
        self.targetPosition = self.waypoints.pop(0)
        print('target position', self.targetPosition)
        self.cmd_position(self.targetPosition[0], self.targetPosition[1], self.targetPosition[2], self.targetPosition[3])

    def landingTransition(self):
        self.flightState = States.LANDING
        print("landing transition")
        self.land()

    def disarmingTransition(self):
        self.flightState = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manualTransition(self):
        self.flightState = States.MANUAL
        print("manual transition")
        self.stop()
        self.inMission = False

    def sendWaypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def point(self, p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

    def collinearityCheck(self, p1, p2, p3, epsilon=1e-6):   
        m = np.concatenate((p1, p2, p3), 0)
        det = np.linalg.det(m)
        return abs(det) < epsilon

    def getDestinationCoordinates(self, origin, distance, bearing, targetPosition): #      
        lat1 = radians(origin[1])
        long1 = radians(origin[0])
        d=distance
        b=radians(bearing)
        R=6371 #km

        lat2 = asin(sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(b))
        long2 = long1 + atan2(sin(b)*sin(d/R)*cos(lat1), cos(d/R)*sin(lat1)*sin(lat2)) #
        #long2 = long1 + atan2(cos(d/R)-sin(lat1)*sin(lat2), sin(b)*sin(d/R)*cos(lat1))
        lat = degrees(lat2)
        long = degrees(long2)
        print("Origin Lat {}, long {}".format(origin[1], origin[0]))
        print("Destination Lat {}, long {}".format(lat, long))

        return np.array([long, lat, targetPosition]) #+180


    def prunePath(self, path):
        if path is not None:
            prunedPath = [p for p in path]
            k=0
            while k < len(prunedPath)-2:  #Not including the first and last.
                p1=self.point(prunedPath[k])
                p2=self.point(prunedPath[k+1])
                p3=self.point(prunedPath[k+2])
                #Remove the middle if three points are collinear
                if self.collinearityCheck(p1,p2,p3):
                    prunedPath.remove(prunedPath[k+1])
                else:
                    k=k+1
        else:
            prunedPath = path
        
        return prunedPath

    def planPath(self):
        self.flightState = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.targetPosition[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        with open("colliders.csv", 'r') as f:
            reader = csv.reader(f)
            values = next(reader) #read the first line.
            lat0, lon0 = float(values[0][4:]), float(values[1][4:]) #Please note there was an extra space in front of the lon0 value in the csv file, I removed it to make the code read uniformly.
            #print("Origin lat {0}, long {0}", lat0, lon0) , should return lat0 = 37.792480, lon0 = -122.397450


        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0,lat0,0) 
        

        # TODO: retrieve current global position
        # TODO: convert to current local position using global_to_local()
        localNorth, localEast, localDown = global_to_local(self.global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, northOffset, eastOffset = createGrid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        #plt.imshow(grid, origin='lower') 

        #plt.xlabel('EAST')
        #plt.ylabel('NORTH')
        #plt.show()

        ##print("North offset = {0}, east offset = {1}".format(northOffset, eastOffset))
        # Define starting point on the grid (this is just grid center)
        gridStart = (int(np.ceil(localNorth)-northOffset), int(np.ceil(localEast)-eastOffset))
        ##print("Home position: {}, {}".format(self.global_home[0], self.global_home[1]))
        # TODO: convert start position to current position rather than map center
        
        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert
        print("Destination address: {0}".format(self.destAddr))
        self.goalPosition = self.getDestinationCoordinates(self.global_home, self.destAddr[0], self.destAddr[1], self.targetPosition[2]) #Distance is in km. Set goal to be 100m from the origin. 0.1, 90,
        #print(self.goalPosition)
        ##print("Destination lat {}, long {}", self.goalPosition[1], self.goalPosition[0])
        goalNorth, goalEast, goalAlt = global_to_local(self.goalPosition, self.global_home)
        print("Goal North, East, Alt: {}, {}, {}".format(goalNorth, goalEast, goalAlt))
        gridGoal = (int(np.ceil(goalNorth-northOffset)) + 10, int(np.ceil(goalEast-eastOffset)) + 10)
        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        #gridGoal=(750., 370.) #For testing purposes, since I am getting an error when I try my destionation coordinate function.
        print('Local Start and Goal: ', gridStart, gridGoal)
        path, _ = aStar(grid, heuristic, gridStart, gridGoal)
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        prunedPath = self.prunePath(path)
        print(prunedPath)
        # Convert path to waypoints
        waypoints = [[p[0] + northOffset, p[1] + eastOffset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.sendWaypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")

    #Pass in customized destination coordinates.
    parser.add_argument('--goalDist', type=float, default=0.1, help="The distance from the home position to the goal.")
    parser.add_argument('--goalBearing', type=float, default=90.0, help="The bearing from the home to the goal.")
    #parser.add_argument('--goalAlt', type=str, help="The desired altitude to fly at to the goal.")

    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    destinationAddress = np.fromstring(f'{args.goalDist},{args.goalBearing}', dtype='Float64', sep=',') #,{args.goalAlt}
    drone = MotionPlanning(conn, destAddr=destinationAddress)
    time.sleep(1)

    drone.start()