import math

class Vector:
    def __init__(self, x: float | int, y: float | int, z: float | int = 0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, a: float | int):
        return Vector(a * self.x, a * self.y, a * self.z)
    
    def __len__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def mult(self, a: float | int):
        return Vector(a * self.x, a * self.y, a * self.z)

    def mag(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def magSq(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def normalise(self):
        if self.mag() != 0:
            return self.mult(1 / self.mag())

    def setMag(self, m: float | int):
        return Vector(self.x / self.mag(), self.y / self.mag(), self.z / self.mag()).mult(m)

    def dir(self): # z = 0
        return(math.atan2(self.y, self.x))

    def setDir(self, t: float | int): # z = 0
        '''The angle is measured in radians, clockwise from the positive x-axis.'''
        return Vector(self.mag() * math.cos(t), self.mag() * math.sin(t), self.z)

    def rotate(self, t: float | int): # z = 0
        '''The angle is measured in radians and the clockwise direction is taken to be positive.'''
        return Vector(self.mag() * math.cos(self.dir() + t), self.mag() * math.sin(self.dir() + t), self.z)

def add(a: Vector, b: Vector):
    return Vector(a.x + b.x, a.y + b.y, a.z + b.z)

def sub(a: Vector, b: Vector):
    return Vector(a.x - b.x, a.y - b.y, a.z - b.z)

def dot(a: Vector, b: Vector):
    return a.x * b.x + a.y * b.y + a.z * b.z

def cross(a: Vector, b: Vector):
    return Vector(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

def distBetween(a: Vector, b: Vector):
    return Vector(a.x - b.x, a.y - b.y, a.z - b.z).mag()

def angBetween(a: Vector, b: Vector):
    '''The returned angle will be in radians.'''
    return math.acos(dot(a, b) / (a.mag() * b.mag()))
