import math

import numpy
import sympy
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


circle_radius = 0.75

def draw_circle(X, Y):
    CX = [X + circle_radius * math.cos(i / 10) for i in range(0, 64)]
    CY = [Y + circle_radius * math.sin(i / 10) for i in range(0, 64)]

    return CX, CY


def animation(i):
    beam1.set_data([-4, -4 + X_ab[i]], [0, Y_ab[i]])
    beam2.set_data([4, 4 + X_ab[i]], [0, Y_ab[i]])
    beam3.set_data([-4 + X_ab[i], 4 + X_ab[i]], [Y_ab[i], Y_ab[i]])
    beam4.set_data([X_ab[i], X_ab[i] + X_o[i]], [Y_ab[i], Y_ab[i] + Y_o[i]])
    circle.set_data(*draw_circle(X_ab[i] + X_o[i], Y_ab[i] + Y_o[i]))

    return beam1, beam2, beam3, beam4, circle

def formY(y, t, f_om_theta, f_om_phi):
    y1, y2, y3, y4 = y
    dydt = [y4, y3, f_om_theta(y1, y2, y3, y4), f_om_phi(y1, y2, y3, y4)]
    return dydt


m_2 = 4
m_1 = 2
g = 10
l = 4
b = 0.5
k = 0.2
R = 0.75

t = sympy.Symbol('t')
phi = sympy.Function('phi')(t)
theta = sympy.Function('theta')(t)

om = sympy.Function('om')(t)
om_2 = sympy.Function('om_2')(t)

E_ab = (m_2 * om**2 * l**2) / 2

Vx_c = om * l * sympy.sin(phi)
Vy_c = om * l * sympy.cos(phi)

Vx_o = om_2 * sympy.sin(theta) * b
Vy_o = om_2 * sympy.cos(theta) * b

J_cir = (m_1 * R**2) / 2
E_cir = m_1 * ((Vx_o + Vx_c)**2 + (Vy_o + Vy_c)**2) / 2 + (J_cir * om_2**2)/2


Ekin = E_ab + E_cir

Pi1 = m_2 * g * (l * (1 - sympy.cos(phi)) + b)
Pi2 = m_1 * g * (l * (1 - sympy.cos(phi)) + b * (1 - sympy.cos(theta)))
Epot = Pi1 + Pi2

M = - k * om_2
L = Ekin - Epot

ur1 = sympy.diff(sympy.diff(L, om_2), t) - sympy.diff(L, theta) - M
ur2 = sympy.diff(sympy.diff(L, om), t) - sympy.diff(L, phi)

# вычисление вторых производных (DV/dt и dom/dt) с использованием метода Крамера
a11 = ur1.coeff(sympy.diff(om_2, t), 1)
a12 = ur1.coeff(sympy.diff(om, t), 1)
a21 = ur2.coeff(sympy.diff(om_2, t), 1)
a22 = ur2.coeff(sympy.diff(om, t), 1)

b1 = -(ur1.coeff(sympy.diff(om_2, t), 0)).coeff(sympy.diff(om, t), 0).subs([(sympy.diff(theta, t), om_2), (sympy.diff(phi, t), om)])
b2 = -(ur2.coeff(sympy.diff(om_2, t), 0)).coeff(sympy.diff(om, t), 0).subs([(sympy.diff(theta, t), om_2), (sympy.diff(phi, t), om)])

detA = a11*a22-a12*a21
detA1 = b1*a22-b2*a21
detA2 = a11*b2-b1*a21

dom_thetadt = detA2/detA
dom_phidt = detA1/detA


iterations = 500

# Построение системы д/у
T = numpy.linspace(0, 10, iterations)

f_om_theta = sympy.lambdify([phi, theta, om_2, om], dom_thetadt, "numpy")
f_om_phi = sympy.lambdify([phi, theta, om_2, om], dom_phidt, "numpy")
y0 = [0.75, 0.75, 15, 1]
sol = odeint(formY, y0, T, args=(f_om_phi, f_om_theta))

X_ab_func = sympy.lambdify(phi, sympy.sin(phi) * l)
Y_ab_func = sympy.lambdify(phi, -sympy.cos(phi) * l)

X_o_func = sympy.lambdify(theta, sympy.sin(theta) * b)
Y_o_func = sympy.lambdify(theta, -sympy.cos(theta) * b)


X_ab = X_ab_func(sol[:, 0])
Y_ab = Y_ab_func(sol[:, 0])

Phi = sol[:, 1]
X_o = X_o_func(sol[:, 1])
Y_o = Y_o_func(sol[:, 1])

fig = plt.figure(figsize = (17, 8))

ax1 = fig.add_subplot(2, 1, 1)
ax1.axis('equal')
ax1.set(xlim = [-10, 10], ylim = [-10, 4])

ax2 = fig.add_subplot(4, 1, 3)
ax2.plot(T, sol[:, 3])

ax2.set_xlabel('T')
ax2.set_ylabel('om phi')

ax3 = fig.add_subplot(4, 1, 4)
ax3.plot(T, sol[:, 2])

ax3.set_xlabel('T')
ax3.set_ylabel('om theta')

beam1, = ax1.plot([-4, -4 + X_ab[0]], [0, Y_ab[0]], 'black')
beam2, = ax1.plot([4, 4 + X_ab[0]], [0, Y_ab[0]], 'black')
beam3, = ax1.plot([-4 + X_ab[0], 4 + X_ab[0]], [Y_ab[0], Y_ab[0]], 'black')
beam4, = ax1.plot([X_ab[0], X_ab[0] + X_o[0]], [Y_ab[0], Y_ab[0] + Y_o[0]], 'black')
circle, = ax1.plot(*draw_circle(X_ab[0] + X_o[0], Y_ab[0] + Y_o[0]), 'black')

anim = FuncAnimation(fig, animation, frames=iterations - 1, interval=0, blit=True)
plt.show()