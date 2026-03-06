from robodk.robolink import *
from robodk.robomath import *
import builtins

PRINT_CONSOLE = False
SHOW_ERROR = True

def getRDK():
    return Robolink()

def print(mensaje, popup = True):
    if PRINT_CONSOLE:
        builtins.print(mensaje)
    else:
        getRDK().ShowMessage(mensaje, popup)

def addFrame(nombre, pose=None):
    rdk = getRDK()
    frame = rdk.AddFrame(nombre)
    if pose is None:
        pose = transl(0, 0, 0)
    else:
        x = pose[0]
        y = pose[1]
        z = pose[2]
        pose = transl(x, y, z)
    frame.setPose(pose)
    return frame

def getRobot(nombre):
    rdk = getRDK()
    robot = rdk.Item(nombre, ITEM_TYPE_ROBOT)
    if not robot.Valid():
        print(f"Error: Robot {nombre} no encontrado.", SHOW_ERROR)
        return None
    return robot

def getFrame(nombre):
    rdk = getRDK()
    frame = rdk.Item(nombre, ITEM_TYPE_FRAME)
    if not frame.Valid():
        print(f"Error: Sistema de referencia {frame} no encontrado, creandolo", SHOW_ERROR)
        frame = addFrame(nombre)
        return frame
    return frame

def getItem(nombre, tipo=None):
    rdk = getRDK()
    item = rdk.Item(nombre, tipo) if tipo else rdk.Item(nombre)
    if not item.Valid():
        print(f"Error: Objeto {nombre} no encontrado.", SHOW_ERROR)
        return None
    return item

def createOrUpdateTarget(nombre, robot, pose):
    """
    Crea (o reutiliza) un target con `nombre` asociado al `robot` y
    le asigna la `pose` indicada (objeto Mat de robomath).
    Devuelve el item target.
    """
    rdk = getRDK()
    target = rdk.Item(nombre, ITEM_TYPE_TARGET)
    if not target.Valid():
        target = rdk.AddTarget(nombre, 0, robot)
    target.setAsCartesianTarget()
    target.setPose(pose)
    return target
