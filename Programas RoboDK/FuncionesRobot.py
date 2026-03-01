from FuncionesBase import *

SHOW_ERROR = True

def setSpeed(robot, velLineal, accelLineal, velAngular, accelAngular):
    if robot:
        robot.setSpeed(velLineal, velAngular, accelLineal, accelAngular)
        return True
    else:
        return False

def moveTo(robot, obj, tipoMov = "MoveJ"):
    if robot and obj:
        if tipoMov == "MoveJ":
            robot.MoveJ(obj)
        elif tipoMov == "MoveL":
            robot.MoveL(obj)
        return True
    else:
        return None

def setPose(robot, pose):
    if robot:
        robot.setJoints(pose)
        return True
    else:
        return None
