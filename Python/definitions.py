import numpy as np
import decimal
from decimal import Decimal

class CDecimal:
    def __init__(self, real, imag=0.0):
        self.real = Decimal(real)
        self.imag = Decimal(imag)

    def __add__(self, other):
        if isinstance(other, CDecimal):
            return CDecimal(self.real + other.real, self.imag + other.imag)
        if isinstance(other, complex):
            return CDecimal(self.real + other.real, self.imag + other.imag)
        assert isinstance(other, int) or isinstance(other, float) or \
               isinstance(other, Decimal)
        return CDecimal(self.real + Decimal(other), self.imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, CDecimal):
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + other.real * self.imag
            return CDecimal(real, imag)
        if isinstance(other, complex):
            real = self.real * Decimal(other.real) - self.imag * Decimal(other.imag)
            imag = self.real * Decimal(other.imag) + Decimal(other.real) * self.imag
            return CDecimal(real, imag)
        if isinstance(other, int) or isinstance(other, float) or \
           isinstance(other, Decimal):
            return CDecimal(self.real * Decimal(other), self.imag * Decimal(other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return CDecimal(-self.real, -self.imag)

    def __sub__(self, other):
        if isinstance(other, CDecimal):
            return CDecimal(self.real - other.real, self.imag - other.imag)
        if isinstance(other, complex):
            return CDecimal(self.real - other.real, self.imag - other.imag)
        assert isinstance(other, int) or isinstance(other, float) or \
               isinstance(other, Decimal)
        return CDecimal(self.real - Decimal(other), self.imag)

    def __rsub__(self, other):
        return -(self.__sub__(other))

    def __truediv__(self, other):
        if isinstance(other, CDecimal):
            real = self.real * other.real + self.imag * other.imag
            imag = -self.real * other.imag + other.real + self.imag
            denom = other.real * other.real + other.imag * other.imag
            return CDecimal(real / denom, imag / denom)
        if isinstance(other, complex):
            real = self.real * other.real + self.imag * other.imag
            imag = -self.real * other.imag + other.real + self.imag
            denom = other.real * other.real + other.imag * other.imag
            return CDecimal(real / denom, imag / denom)
        assert isinstance(other, int) or isinstance(other, float) or \
               isinstance(other, Decimal)
        return CDecimal(self.real / other, self.imag / other)

    def conjugate(self):
        return CDecimal(self.real, -self.imag)


def cos(x):
    """Return the cosine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(cos(Decimal('0.5')))
    0.8775825618903727161162815826
    >>> print(cos(0.5))
    0.87758256189
    >>> print(cos(0.5+0j))
    (0.87758256189+0j)

    """
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 0, 0, 1, 1, 1, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s

def sin(x):
    """Return the sine of x as measured in radians.

    The Taylor series approximation works best for a small value of x.
    For larger values, first compute x = x % (2 * pi).

    >>> print(sin(Decimal('0.5')))
    0.4794255386042030002732879352
    >>> print(sin(0.5))
    0.479425538604
    >>> print(sin(0.5+0j))
    (0.479425538604+0j)

    """
    decimal.getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i-1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    decimal.getcontext().prec -= 2
    return +s


def exp_imag(angle):
    a = Decimal(angle)
    return CDecimal(cos(a), sin(a))


# Pauli matrices and projector |+><+|
x = np.array([[0, 1], [1, 0]])
y = np.array([[0, -1j], [1j, 0]])
z = np.array([[1, 0], [0, -1]])
one = np.array([[1, 0], [0, 1]])
projx = (one + x) / 2

def pi():
    decimal.getcontext().prec += 2
    three = Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n+na, na+8
        d, da = d+da, da+32
        t = (t * n) / d
        s += t
    decimal.getcontext().prec -= 2
    return +s

sqrt_2 = np.sqrt(Decimal(2))
sqrt_8 = np.sqrt(Decimal(8))
decimal_pi = pi()


# Density matrices of pure |+> states, pure magic states and pure CCZ states
plusstate = np.dot(np.array([[1 / sqrt_2], [1 / sqrt_2]]), np.array([[1 / sqrt_2, 1 / sqrt_2]]))
magicstate = np.dot(np.array([[CDecimal(1 / sqrt_2)], [exp_imag(decimal_pi / 4) * 1 / sqrt_2]]),
                    np.array([[1 / sqrt_2, exp_imag(-decimal_pi / 4) * 1 / sqrt_2]]))
CCZstate = np.dot(np.array([[1 / sqrt_8], [1 / sqrt_8], [1 / sqrt_8], [1 / sqrt_8], [1 / sqrt_8],
                            [1 / sqrt_8], [1 / sqrt_8], [-1 / sqrt_8]]), np.array([[1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                1 / sqrt_8,
                                                                                                -1 / sqrt_8]]))


# Computes the tensor product of a list of matrices
def kronecker_product(matrices):
    res = np.kron(matrices[0], matrices[1])
    for i in matrices[2:]:
        res = np.kron(res, i)
    return res


# Density matrices of 5, 7 and 4 |+> states
init5qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate, plusstate])
init7qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate, plusstate, plusstate, plusstate])
init4qubit = kronecker_product([plusstate, plusstate, plusstate, plusstate])

# Density matrices corresponding to the ideal output state of 15-to-1, 20-to-4 and 8-to-CCZ
ideal15to1 = kronecker_product([magicstate, plusstate, plusstate, plusstate, plusstate])
ideal20to4 = kronecker_product([magicstate, magicstate, magicstate, magicstate, plusstate, plusstate, plusstate])
ideal8toCCZ = kronecker_product([CCZstate, plusstate])


# Pauli product rotation e^(iP*phi), where the Pauli product P is specified by 'axis' and phi is the rotation angle
def pauli_rot(axis, angle):
    id = np.vectorize(Decimal)(np.eye(2 ** len(axis)))
    return cos(angle) * id + CDecimal(0, sin(angle)) * kronecker_product(axis)


# Applies a pi/8 Pauli product rotation specified by 'axis' with probability 1-p1-p2-p3
# A P_(pi/2) / P_(-pi/4) / P_(pi/4) error occurs with probability p1 / p2 / p3
def apply_rot(state, axis, p1, p2, p3):
    return (1 - p1 - p2 - p3) * np.dot(np.dot(pauli_rot(axis, decimal_pi / 8), state),
                                       pauli_rot(axis, decimal_pi / 8).conj().transpose()) \
           + p1 * np.dot(np.dot(pauli_rot(axis, 5 * decimal_pi / 8), state),
                         pauli_rot(axis, 5 * decimal_pi / 8).conj().transpose()) \
           + p2 * np.dot(np.dot(pauli_rot(axis, -1 * decimal_pi / 8), state),
                         pauli_rot(axis, -1 * decimal_pi / 8).conj().transpose()) \
           + p3 * np.dot(np.dot(pauli_rot(axis, 3 * decimal_pi / 8), state),
                         pauli_rot(axis, 3 * decimal_pi / 8).conj().transpose())


# Applies a Pauli operator to a state with probability p
def apply_pauli(state, pauli, p):
    return (1 - p) * state + p * np.dot(np.dot(kronecker_product(pauli), state), kronecker_product(pauli))


# Estimate of the logical error rate of a surface-code patch with code distance d and circuit-level error rate pphys
def plog(pphys, d):
    return 1 / 10 * (100 * pphys) ** ((d + 1) / 2)


# For the 8-to-CCZ protocol, applies X/Z storage errors to qubits 1-4 with probabilities p1-p4
def storage_x_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [x, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one], p2)
    res = apply_pauli(res, [one, one, x, one], p3)
    res = apply_pauli(res, [one, one, one, x], p4)
    return res


def storage_z_4(state, p1, p2, p3, p4):
    res = apply_pauli(state, [z, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one], p2)
    res = apply_pauli(res, [one, one, z, one], p3)
    res = apply_pauli(res, [one, one, one, z], p4)
    return res


# For the 15-to-1 protocol, applies X/Z storage errors to qubits 1-5 with probabilities p1-p5
def storage_x_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [x, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one], p4)
    res = apply_pauli(res, [one, one, one, one, x], p5)
    return res


def storage_z_5(state, p1, p2, p3, p4, p5):
    res = apply_pauli(state, [z, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one], p4)
    res = apply_pauli(res, [one, one, one, one, z], p5)
    return res


# For the 20-to-4 protocol, applies X/Z storage errors to qubits 1-7 with probabilities p1-p7
def storage_x_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [x, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, x, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, x, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, x, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, x, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, x, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, x], p7)
    return res


def storage_z_7(state, p1, p2, p3, p4, p5, p6, p7):
    res = apply_pauli(state, [z, one, one, one, one, one, one], p1)
    res = apply_pauli(res, [one, z, one, one, one, one, one], p2)
    res = apply_pauli(res, [one, one, z, one, one, one, one], p3)
    res = apply_pauli(res, [one, one, one, z, one, one, one], p4)
    res = apply_pauli(res, [one, one, one, one, z, one, one], p5)
    res = apply_pauli(res, [one, one, one, one, one, z, one], p6)
    res = apply_pauli(res, [one, one, one, one, one, one, z], p7)
    return res
