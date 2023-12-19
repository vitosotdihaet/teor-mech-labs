import math

import numpy
import sympy
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def animate(i):
    P.set_data(Xs[i], Ys[i])

    VVec.set_data([Xs[i], Xs[i] + Xs_velocity[i]], [Ys[i], Ys[i] + Ys_velocity[i]])

    RVecArrowX, RVecArrowY = rotation2D(ArrowX, ArrowY, math.atan2(Ys_velocity[i], Xs_velocity[i]))
    VVecArrow.set_data(RVecArrowX + Xs[i] + Xs_velocity[i], RVecArrowY + Ys[i] + Ys_velocity[i])


    AVec.set_data([Xs[i], Xs[i] + Xs_acceleration[i]], [Ys[i], Ys[i] + Ys_acceleration[i]])

    RVecArrowX_A, RVecArrowY_A = rotation2D(ArrowX, ArrowY, math.atan2(Ys_acceleration[i], Xs_acceleration[i]))
    AVecArrow.set_data(RVecArrowX_A + Xs[i] + Xs_acceleration[i], RVecArrowY_A + Ys[i] + Ys_acceleration[i])


    RVec.set_data([0, Xs[i]], [0, Ys[i]])
    
    RVecArrowX_R, RVecArrowY_R = rotation2D(ArrowX, ArrowY, math.atan2(Ys[i], Xs[i]))
    RVecArrow.set_data(RVecArrowX_R + Xs[i], RVecArrowY_R + Ys[i])

    return P, VVec, VVecArrow, AVec, AVecArrow, RVec, RVecArrow


def rotation2D(x, y, angle):
    Rx = x * numpy.cos(angle) - y * numpy.sin(angle)
    Ry = x * numpy.sin(angle) + y * numpy.cos(angle)
    return Rx, Ry


t = sympy.Symbol('t')

r = 2 + sympy.sin(6 * t)
phi = 7 * t + 1.2 * sympy.cos(6 * t)

x = r * sympy.cos(phi)
y = r * sympy.sin(phi)

x_velocity = sympy.diff(x, t)
y_velocity = sympy.diff(y, t)

x_acceleration = sympy.diff(x_velocity, t) 
y_acceleration = sympy.diff(y_velocity, t) 


T = numpy.linspace(1, 10, 2000)

Xs = numpy.zeros_like(T)
Ys = numpy.zeros_like(T)

Xs_velocity = numpy.zeros_like(T)
Ys_velocity = numpy.zeros_like(T)

Xs_acceleration = numpy.zeros_like(T)
Ys_acceleration = numpy.zeros_like(T)


for i in numpy.arange(len(T)):
    Xs[i] = sympy.Subs(x, t, T[i])
    Ys[i] = sympy.Subs(y, t, T[i])

    Xs_velocity[i] = sympy.Subs(x_velocity, t, T[i])
    Ys_velocity[i] = sympy.Subs(y_velocity, t, T[i])

    Xs_acceleration[i] = sympy.Subs(x_acceleration, t, T[i])
    Ys_acceleration[i] = sympy.Subs(y_acceleration, t, T[i])


fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim = [-3.5, 3.5], ylim = [-3.5, 3.5])
ax1.plot(Xs, Ys)

P, = ax1.plot(Xs[0], Ys[0], marker = 'o')

ArrowX = numpy.array([-0.2, 0, -0.2])
ArrowY = numpy.array([0.1, 0, -0.1])

VVec, = ax1.plot([Xs[0], Xs[0] + Xs_velocity[0]], [Ys[0], Ys[0] + Ys_velocity[0]], 'r')
RVecArrowX, RVecArrowY = rotation2D(ArrowX, ArrowY, math.atan2(Ys_velocity[0], Xs_velocity[0]))
VVecArrow, = ax1.plot(RVecArrowX + Xs_velocity[0] + Xs[0], RVecArrowY + Ys_velocity[0] + Ys[0], 'r')

RVec, = ax1.plot([0, Xs[0]], [0, Ys[0]], 'b')
RVecArrowX_R, RVecArrowY_R = rotation2D(ArrowX, ArrowY, math.atan2(Ys[0], Xs[0]))
RVecArrow, = ax1.plot(RVecArrowX_R + Xs_velocity[0] + Xs[0], RVecArrowY_R + Ys_velocity[0] + Ys[0], 'b')

AVec, = ax1.plot([Xs[0], Xs[0] + Xs_acceleration[0]], [Ys[0], Ys[0] + Ys_acceleration[0]], 'g')
RVecArrowX_A, RVecArrowY_A = rotation2D(ArrowX, ArrowY, math.atan2(Ys_acceleration[0], Xs_acceleration[0]))
AVecArrow, = ax1.plot(RVecArrowX_A + Xs[0], RVecArrowY_A + Ys[0], 'g')


animation = FuncAnimation(fig, animate, frames = 1000, interval = 10, blit = True)

plt.show()