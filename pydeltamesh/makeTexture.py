from enum import Enum

import numpy as _np


class OpCode(Enum):
    Add = 1
    Subtract = 2
    Multiply = 3
    Divide = 4
    Sin = 5
    Cos = 6
    Tan = 7
    Sqrt = 8
    Pow = 9
    Exp = 10
    Log = 11
    Mod = 12
    Abs = 13
    Sign = 14
    Min = 15
    Max = 16
    Clamp = 17
    Ceil = 18
    Floor = 19
    Round = 20
    Step = 21
    Smoothstep = 22
    Bias = 23
    Gain = 24


op = {
    OpCode.Add: lambda a, b: a + b,
    OpCode.Subtract: lambda a, b: a - b,
    OpCode.Multiply: lambda a, b: a * b,
    OpCode.Divide: lambda a, b: a / b,
    OpCode.Sin: lambda a, _: _np.sin(a),
    OpCode.Cos: lambda a, _: _np.cos(a),
    OpCode.Tan: lambda a, _: _np.tan(a),
    OpCode.Sqrt: lambda a, _: _np.sqrt(a),
    OpCode.Pow: lambda a, b: a ** b,
    OpCode.Exp: lambda a, _: _np.exp(a),
    OpCode.Log: lambda a, b: _np.log(a) / _np.log(b),
    OpCode.Mod: lambda a, b: a % (_np.sign(a) * b),
    OpCode.Abs: lambda a, _: _np.abs(a),
    OpCode.Sign: lambda a, _: _np.sign(a),
    OpCode.Min: lambda a, b: _np.minimum(a, b),
    OpCode.Max: lambda a, b: _np.maximum(a, b),
    OpCode.Clamp: lambda a, _: _np.clip(a, 0, 1),
    OpCode.Ceil: lambda a, _: _np.ceil(a),
    OpCode.Floor: lambda a, _: _np.floor(a),
    OpCode.Round: lambda a, _: _np.round(a),
    OpCode.Step: lambda a, b: (a <= b).astype(_np.float32),
    OpCode.Smoothstep: lambda a, _: _np.clip(3 * a**2 - 2 * a**3, 0, 1),
    #OpCode.Bias: lambda a, b: a + b,
    #OpCode.Gain: lambda a, b: a + b,
}


class Node(object):
    __next_id = [1]

    def __init__(self):
        self.id = self.__next_id[0]
        self.__next_id[0] += 1

    def __add__(self, other):
        return MathFun(OpCode.Add, self, other)

    def __radd__(self, other):
        return MathFun(OpCode.Add, other, self)

    def __sub__(self, other):
        return MathFun(OpCode.Subtract, self, other)

    def __rsub__(self, other):
        return MathFun(OpCode.Subtract, other, self)

    def __mul__(self, other):
        return MathFun(OpCode.Multiply, self, other)

    def __rmul__(self, other):
        return MathFun(OpCode.Multiply, other, self)

    def __truediv__(self, other):
        return MathFun(OpCode.Divide, self, other)

    def __rtruediv__(self, other):
        return MathFun(OpCode.Divide, other, self)

    def __pow__(self, other):
        return MathFun(OpCode.Pow, self, other)

    def __rpow__(self, other):
        return MathFun(OpCode.Pow, other, self)

    def __mod__(self, other):
        return MathFun(OpCode.Mod, self, other)

    def __rmod__(self, other):
        return MathFun(OpCode.Mod, other, self)

    def __le__(self, other):
        return MathFun(OpCode.Step, self, other)

    def __ge__(self, other):
        return MathFun(OpCode.Step, other, self)

    def __eq__(self, other):
        return (self <= other) * (self >= other)

    def inv(self):
        return MathFun(OpCode.Subtract, 1, self)

    def __lt__(self, other):
        return (self >= other).inv()

    def __gt__(self, other):
        return (self <= other).inv()

    def __ne__(self, other):
        return (self == other).inv()

    def sin(self):
        return MathFun(OpCode.Sin, self, 0)

    def cos(self):
        return MathFun(OpCode.Cos, self, 0)

    def tan(self):
        return MathFun(OpCode.Tan, self, 0)

    def sqrt(self):
        return MathFun(OpCode.Sqrt, self, 0)

    def exp(self):
        return MathFun(OpCode.Exp, self, 0)

    def log(self, other):
        return MathFun(OpCode.Log, self, other)

    def abs(self):
        return MathFun(OpCode.Abs, self, 0)

    def sign(self):
        return MathFun(OpCode.Sign, self, 0)

    def min(self, other):
        return MathFun(OpCode.Min, self, other)

    def max(self, other):
        return MathFun(OpCode.Max, self, other)

    def clamp(self):
        return MathFun(OpCode.Clamp, self, 0)

    def ceil(self):
        return MathFun(OpCode.Ceil, self, 0)

    def floor(self):
        return MathFun(OpCode.Floor, self, 0)

    def round(self):
        return MathFun(OpCode.Round, self, 0)

    def smoothstep(self):
        return MathFun(OpCode.Smoothstep, self, 0)


class U(Node):
    def __init__(self, n=512):
        Node.__init__(self)

        self.data = _np.outer(
            _np.full(n, 1.0),
            _np.arange(0.0, 1.0, 1.0 / n)
        )


class V(Node):
    def __init__(self, n=512):
        Node.__init__(self)

        self.data = _np.outer(
            _np.flip(_np.arange(0.0, 1.0, 1.0 / n)),
            _np.full(n, 1.0)
        )


class MathFun(Node):
    def __init__(self, opcode, val1, val2):
        Node.__init__(self)

        v1 = val1.data if isinstance(val1, Node) else val1
        v2 = val2.data if isinstance(val2, Node) else val2

        self.data = op[opcode](v1, v2)
        self.data[~_np.isfinite(self.data)] = 0.0


if __name__ == "__main__":
    from PIL import Image

    a = ((U() - 0.5)**2 + (V() - 0.5)**2).sqrt() < 0.5
    b = (U() * 10) % 1 < 0.5
    c = U() != V()
    d = ((U() * 4) % 1).smoothstep()

    out = d.data

    Image.fromarray(out * 256).show()
