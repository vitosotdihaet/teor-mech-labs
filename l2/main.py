import math

import numpy
import sympy
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


circle_radius = 0.75

def draw_circle(X, Y):
    CX = [X + circle_radius * math.cos(i / 10) for i in range(0, 64)]
    CY = [Y + circle_radius * math.sin(i / 10) for i in range(0, 64)]

    return CX, CY


def animation(i):
    beam1.set_data([-4, -4 + XA[i]], [0, YA[i]])
    beam2.set_data([4, 4 + XA[i]], [0, YA[i]])
    beam3.set_data([-4 + XA[i], 4 + XA[i]], [YA[i], YA[i]])
    beam4.set_data([XA[i], XA[i] + XO[i]], [YA[i], YA[i] + YO[i]])
    circle.set_data(*draw_circle(XA[i] + XO[i], YA[i] + YO[i]))

    return beam1, beam2, beam3, beam4, circle

t = sympy.Symbol('t')

phi = sympy.sin(t) * sympy.cos(t) + math.pi/4
theta = 3 * t


beam1_length = 4
beam2_length = 1.5

# speed and acceleration of point A
A_x_velocity = sympy.diff(sympy.sin(phi) * beam1_length, t)
A_y_velocity = sympy.diff(sympy.cos(phi) * beam1_length, t)

A_velocity = (A_x_velocity**2 + A_y_velocity**2)**0.5
A_acceleration = (sympy.diff(A_x_velocity, t)**2 + sympy.diff(A_y_velocity, t)**2)**0.5

# speed and acceleration of point C
C_x_velocity = sympy.diff(sympy.sin(theta) * beam2_length, t) + A_x_velocity
C_y_velocity = sympy.diff(sympy.cos(theta) * beam2_length, t) + A_y_velocity

C_velocity = (C_x_velocity**2 + C_y_velocity**2)**0.5
C_acceleration = (sympy.diff(C_x_velocity, t)**2 + sympy.diff(C_y_velocity, t)**2)**0.5


iterations = 1000
T = numpy.linspace(0, 2*math.pi, num=iterations)

XO = numpy.zeros_like(T)
YO = numpy.zeros_like(T)

XA = numpy.zeros_like(T)
YA = numpy.zeros_like(T)

VA = numpy.zeros_like(T)
WA = numpy.zeros_like(T)

VC = numpy.zeros_like(T)
WC = numpy.zeros_like(T)

for i in numpy.arange(iterations):
    XO[i] = sympy.Subs(beam2_length * sympy.cos(theta), t, T[i])
    YO[i] = sympy.Subs(beam2_length * sympy.sin(theta), t, T[i])
    
    XA[i] = sympy.Subs(beam1_length * sympy.cos(2 * phi), t, T[i])
    YA[i] = sympy.Subs(-math.sqrt(beam1_length**2 - XA[i]**2), t, T[i])
    
    VA[i] = sympy.Subs(A_velocity, t, T[i])
    WA[i] = sympy.Subs(A_acceleration, t, T[i])

    VC[i] = sympy.Subs(C_velocity, t, T[i])
    WC[i] = sympy.Subs(C_acceleration, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
# ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[-10, 10], ylim=[-10, 10])

# ax2 = fig.add_subplot(4, 2, 2)
# ax2.plot(T, VA)

# ax2.set_xlabel('T')
# ax2.set_ylabel('velocity of A')

# ax3 = fig.add_subplot(4, 2, 4)
# ax3.plot(T, WA)

# ax3.set_xlabel('T')
# ax3.set_ylabel('acceleration of A')

# ax4 = fig.add_subplot(4, 2, 6)
# ax4.plot(T, VC)

# ax4.set_xlabel('T')
# ax4.set_ylabel('velocity of C')

# ax5 = fig.add_subplot(4, 2, 8)
# ax5.plot(T, WC)

# ax5.set_xlabel('T')
# ax5.set_ylabel('acceleration of C')

beam1, = ax1.plot([-4, -4 + XA[0]], [0, YA[0]], 'black')
beam2, = ax1.plot([4, 4 + XA[0]], [0, YA[0]], 'black')
beam3, = ax1.plot([-4 + XA[0], 4 + XA[0]], [YA[0], YA[0]], 'black')
beam4, = ax1.plot([XA[0], XA[0] + XO[0]], [YA[0], YA[0] + YO[0]], 'black')
circle, = ax1.plot(*draw_circle(XA[0] + XO[0], YA[0] + YO[0]), 'black')

anim = FuncAnimation(fig, animation, frames=iterations - 1, interval=0, cache_frame_data=False, blit=True)

plt.show()