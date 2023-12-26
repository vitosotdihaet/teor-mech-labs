import math

import numpy
import sympy
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def draw_circle(X, Y):
    CX = [X + r * math.cos(i / 10) for i in range(0, 64)]
    CY = [Y + r * math.sin(i / 10) for i in range(0, 64)]

    return CX, CY


def animation(i):
    beam1.set_data([-l, -l + X_ab[i]], [0, Y_ab[i]])
    beam2.set_data([l, l + X_ab[i]], [0, Y_ab[i]])
    beam3.set_data([-l + X_ab[i], l + X_ab[i]], [Y_ab[i], Y_ab[i]])
    beam4.set_data([X_ab[i], X_ab[i] + X_o[i]], [Y_ab[i], Y_ab[i] + Y_o[i]])
    circle.set_data(*draw_circle(X_ab[i] + X_o[i], Y_ab[i] + Y_o[i]))

    return beam1, beam2, beam3, beam4, circle

def formY(y, t, f_omega_phi, f_omega_theta):
    y1, y2, y3, y4 = y
    dydt = [y4, y3, f_omega_phi(y1, y2, y3, y4), f_omega_theta(y1, y2, y3, y4)]
    return dydt



g = 10

m_1 = 5
m_2 = 5
l = 1
b = 0.125
k = 0.1
r = 0.25

t = sympy.Symbol('t')
phi = sympy.Function('phi')(t)
theta = sympy.Function('theta')(t)

omega_phi = sympy.Function('omega_phi')(t)
omega_theta = sympy.Function('omega_theta')(t)

E_ab = (m_2 * omega_phi**2 * l**2) / 2

Vx_c = omega_phi * l * sympy.sin(phi)
Vy_c = omega_phi * l * sympy.cos(phi)

Vx_o = omega_theta * sympy.sin(theta) * b
Vy_o = omega_theta * sympy.cos(theta) * b

J_cir = (m_1 * r**2) / 2
E_cir = m_1 * ((Vx_o + Vx_c)**2 + (Vy_o + Vy_c)**2) / 2 + (J_cir * omega_theta**2)/2


Ekin = E_ab + E_cir

Pi1 = m_1 * g * (l * (1 - sympy.cos(phi)) + b * (1 - sympy.cos(theta)))
Pi2 = m_2 * g * (l * (1 - sympy.cos(phi)) + b)
Epot = Pi1 + Pi2

M = - k * omega_theta
L = Ekin - Epot

eq1 = sympy.diff(sympy.diff(L, omega_theta), t) - sympy.diff(L, theta) - M
eq2 = sympy.diff(sympy.diff(L, omega_phi), t) - sympy.diff(L, phi)

# вычисление вторых производных (DV/dt и domega/dt) с использованием метода Крамера
a11 = eq1.coeff(sympy.diff(omega_theta, t), 1)
a12 = eq1.coeff(sympy.diff(omega_phi, t), 1)
a21 = eq2.coeff(sympy.diff(omega_theta, t), 1)
a22 = eq2.coeff(sympy.diff(omega_phi, t), 1)

b1 = -(eq1.coeff(sympy.diff(omega_theta, t), 0)).coeff(sympy.diff(omega_phi, t), 0).subs([(sympy.diff(theta, t), omega_theta), (sympy.diff(phi, t), omega_phi)])
b2 = -(eq2.coeff(sympy.diff(omega_theta, t), 0)).coeff(sympy.diff(omega_phi, t), 0).subs([(sympy.diff(theta, t), omega_theta), (sympy.diff(phi, t), omega_phi)])

detA = a11*a22 - a12*a21
detA1 = b1*a22 - b2*a21
detA2 = a11*b2 - b1*a21

domega_phidt = detA1/detA
domega_thetadt = detA2/detA


iterations = 500

T = numpy.linspace(0, 10, iterations)

# Построение системы д/у
f_omega_phi = sympy.lambdify([phi, theta, omega_theta, omega_phi], domega_phidt, "numpy")
f_omega_theta = sympy.lambdify([phi, theta, omega_theta, omega_phi], domega_thetadt, "numpy")
y0 = [sympy.pi/2, sympy.pi/2, 15, 1]
sol = odeint(formY, y0, T, args=(f_omega_phi, f_omega_theta))

# generate coords of the line
X_ab_func = sympy.lambdify(phi, sympy.sin(phi) * l)
Y_ab_func = sympy.lambdify(phi, -sympy.cos(phi) * l)

X_ab = X_ab_func(sol[:, 0])
Y_ab = Y_ab_func(sol[:, 0])

# generate coords of the circle
X_o_func = sympy.lambdify(theta, sympy.sin(theta) * b)
Y_o_func = sympy.lambdify(theta, -sympy.cos(theta) * b)

X_o = X_o_func(sol[:, 1])
Y_o = Y_o_func(sol[:, 1])


# draw graphs
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax1.axis('equal')
ax1.set(xlim = [-3*l, 3*l], ylim = [-1.5*l, 0.5])

ax2 = fig.add_subplot(4, 1, 3)
ax2.plot(T, sol[:, 3])

ax2.set_xlabel('T')
ax2.set_ylabel('omega phi')

ax3 = fig.add_subplot(4, 1, 4)
ax3.plot(T, sol[:, 2])

ax3.set_xlabel('T')
ax3.set_ylabel('omega theta')

# draw the system
beam1, = ax1.plot([-l, -l + X_ab[0]], [0, Y_ab[0]], 'black')
beam2, = ax1.plot([l, l + X_ab[0]], [0, Y_ab[0]], 'black')
beam3, = ax1.plot([-l + X_ab[0], l + X_ab[0]], [Y_ab[0], Y_ab[0]], 'black')
beam4, = ax1.plot([X_ab[0], X_ab[0] + X_o[0]], [Y_ab[0], Y_ab[0] + Y_o[0]], 'black')
circle, = ax1.plot(*draw_circle(X_ab[0] + X_o[0], Y_ab[0] + Y_o[0]), 'black')

anim = FuncAnimation(fig, animation, frames=iterations - 1, interval=0, blit=True)
plt.show()