from enum import Enum
import numpy as _np

from pydeltamesh.fileio.poserFile import PoserFile


class Op(Enum):
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


def bias(a, b):
    return a**(_np.log(b) / _np.log(0.5))


def gain(a, b):
    return _np.where(
        a < 0.5,
        bias(a * 2, 1 - b) / 2,
        1 - bias((1 - a) * 2, 1 - b) / 2
    )


op = {
    Op.Add: lambda a, b: a + b,
    Op.Subtract: lambda a, b: a - b,
    Op.Multiply: lambda a, b: a * b,
    Op.Divide: lambda a, b: a / b,
    Op.Sin: lambda a, _: _np.sin(a),
    Op.Cos: lambda a, _: _np.cos(a),
    Op.Tan: lambda a, _: _np.tan(a),
    Op.Sqrt: lambda a, _: _np.sqrt(a),
    Op.Pow: lambda a, b: a ** b,
    Op.Exp: lambda a, _: _np.exp(a),
    Op.Log: lambda a, b: _np.log(a) / _np.log(b),
    Op.Mod: lambda a, b: a % (_np.sign(a) * b),
    Op.Abs: lambda a, _: _np.abs(a),
    Op.Sign: lambda a, _: _np.sign(a),
    Op.Min: lambda a, b: _np.minimum(a, b),
    Op.Max: lambda a, b: _np.maximum(a, b),
    Op.Clamp: lambda a, _: _np.clip(a, 0, 1),
    Op.Ceil: lambda a, _: _np.ceil(a),
    Op.Floor: lambda a, _: _np.floor(a),
    Op.Round: lambda a, _: _np.round(a),
    Op.Step: lambda a, b: (a <= b).astype(_np.float32),
    Op.Smoothstep: lambda a, _: _np.clip(3 * a**2 - 2 * a**3, 0, 1),
    Op.Bias: bias,
    Op.Gain: gain,
}


class Node(object):
    __nodes = []

    def __init__(self):
        self.id = len(self.__nodes)
        self.__nodes.append(self)

    def name(self):
        return f'n{self.id:03}'

    def nodes(self):
        return self.__nodes

    def format(self):
        return f"Node_{self.id}"

    def __add__(self, other):
        return MathFun(Op.Add, self, other)

    def __radd__(self, other):
        return MathFun(Op.Add, other, self)

    def __sub__(self, other):
        return MathFun(Op.Subtract, self, other)

    def __rsub__(self, other):
        return MathFun(Op.Subtract, other, self)

    def __mul__(self, other):
        return MathFun(Op.Multiply, self, other)

    def __rmul__(self, other):
        return MathFun(Op.Multiply, other, self)

    def __truediv__(self, other):
        return MathFun(Op.Divide, self, other)

    def __rtruediv__(self, other):
        return MathFun(Op.Divide, other, self)

    def __pow__(self, other):
        return MathFun(Op.Pow, self, other)

    def __rpow__(self, other):
        return MathFun(Op.Pow, other, self)

    def __mod__(self, other):
        return MathFun(Op.Mod, self, other)

    def __rmod__(self, other):
        return MathFun(Op.Mod, other, self)

    def __le__(self, other):
        return MathFun(Op.Step, self, other)

    def __ge__(self, other):
        return MathFun(Op.Step, other, self)

    def __eq__(self, other):
        return (self <= other) * (self >= other)

    def inv(self):
        return MathFun(Op.Subtract, 1, self)

    def __lt__(self, other):
        return (self >= other).inv()

    def __gt__(self, other):
        return (self <= other).inv()

    def __ne__(self, other):
        return (self == other).inv()

    def sin(self):
        return MathFun(Op.Sin, self, 0)

    def cos(self):
        return MathFun(Op.Cos, self, 0)

    def tan(self):
        return MathFun(Op.Tan, self, 0)

    def sqrt(self):
        return MathFun(Op.Sqrt, self, 0)

    def exp(self):
        return MathFun(Op.Exp, self, 0)

    def log(self, other):
        return MathFun(Op.Log, self, other)

    def abs(self):
        return MathFun(Op.Abs, self, 0)

    def sign(self):
        return MathFun(Op.Sign, self, 0)

    def min(self, other):
        return MathFun(Op.Min, self, other)

    def max(self, other):
        return MathFun(Op.Max, self, other)

    def clamp(self):
        return MathFun(Op.Clamp, self, 0)

    def ceil(self):
        return MathFun(Op.Ceil, self, 0)

    def floor(self):
        return MathFun(Op.Floor, self, 0)

    def round(self):
        return MathFun(Op.Round, self, 0)

    def smoothstep(self):
        return MathFun(Op.Smoothstep, self, 0)

    def bias(self, other):
        return MathFun(Op.Bias, self, other)

    def gain(self, other):
        return MathFun(Op.Gain, self, other)


class U(Node):
    def __init__(self, n=512):
        Node.__init__(self)

        self.data = _np.outer(
            _np.full(n, 1.0),
            _np.arange(0.0, 1.0, 1.0 / n)
        )

    def format(self):
        return f"U_{self.id}"
    
    def to_poser(self):
        id = self.id
        name = self.name()

        node = PoserFile(var_node_template.splitlines()).root
        node_spec = next(node.select('node'))
        node_spec.rest = f's _{name}'
        next(node_spec.select('name')).rest = name.title()
        next(node_spec.select('pos')).rest = f'{890 - id * 20} {10 + id * 20}'
        next(node_spec.select('output', 'exposedAs')).rest = f'_{name}_out'

        return node


class V(Node):
    def __init__(self, n=512):
        Node.__init__(self)

        self.data = _np.outer(
            _np.flip(_np.arange(0.0, 1.0, 1.0 / n)),
            _np.full(n, 1.0)
        )

    def format(self):
        return f"V_{self.id}"
    
    def to_poser(self):
        id = self.id
        name = self.name()

        node = PoserFile(var_node_template.splitlines()).root
        node_spec = next(node.select('node'))
        node_spec.rest = f't _{name}'
        next(node_spec.select('name')).rest = name.title()
        next(node_spec.select('pos')).rest = f'{890 - id * 20} {10 + id * 20}'
        next(node_spec.select('output', 'exposedAs')).rest = f'_{name}_out'

        return node


class MathFun(Node):
    def __init__(self, opcode, val1, val2):
        Node.__init__(self)

        v1 = val1.data if isinstance(val1, Node) else val1
        v2 = val2.data if isinstance(val2, Node) else val2

        self.opcode = opcode
        self.inputs = (val1, val2)
        self.data = op[opcode](v1, v2)
        self.data[~_np.isfinite(self.data)] = 0.0

    def format(self):
        id = self.id
        op = self.opcode
        in1 = format_input(self.inputs[0])
        in2 = format_input(self.inputs[1])

        return f"MathFun_{id}: {op}, inputs = ({in1}, {in2})"

    def to_poser(self):
        id = self.id
        name = self.name()
        op = self.opcode
        in1, in2 = self.inputs

        node = PoserFile(math_node_template.splitlines()).root
        node_spec = next(node.select('node'))
        node_spec.rest = f'math_functions _{name}'
        next(node_spec.select('name')).rest = name.title()
        next(node_spec.select('pos')).rest = f'{890 - id * 20} {10 + id * 20}'
        next(node_spec.select('output', 'exposedAs')).rest = f'_{name}_out'

        math_arg, val1, val2 = list(node_spec.select('nodeInput'))
        next(math_arg.select('enumValue')).rest = f'{op.value}'
        next(val1.select('exposedAs')).rest = f'_{name}_in1'
        next(val2.select('exposedAs')).rest = f'_{name}_in2'

        if isinstance(in1, Node):
            next(val1.select('value')).rest = '1 0 100'
            next(val1.select('node')).rest = f'_{in1.name()}:Color'
        else:
            next(val1.select('value')).rest = f'{in1} 0 100'

        if isinstance(in2, Node):
            next(val2.select('value')).rest = '1 0 100'
            next(val2.select('node')).rest = f'_{in2.name()}:Color'
        else:
            next(val2.select('value')).rest = f'{in2} 0 100'

        return node


def format_input(val):
    return f"Node {val.id}" if isinstance(val, Node) else f"Value {val}"


def write_poser_file(fp, name, input_nodes, output_nodes):
    source = PoserFile(file_template.splitlines())
    root = source.root
    compound = next(root.select('actor', 'material', 'shaderTree', 'node'))
    compound.rest = f'compound {name}'
    next(compound.select('name')).rest = name.title()

    tree = next(compound.select('shaderTree'))

    v = output_nodes[0]
    for node in v.nodes():
        if hasattr(node, 'to_poser'):
            next(tree.select('}')).prependSibling(node.to_poser())

    source.writeTo(fp)


file_template = '''{
version
    {
    number 13
    build 581
    }
actor $CURRENT
    {
    material Preview
        {
        $SELECTEDNODES
        shaderTree
            {
            node compound @name
                {
                name @name
                pos 600 600
                compoundOutputsPos 10 10
                compoundInputsPos 230 10
                compoundShowPreview 1
                showPreview 0
                advancedInputsCollapsed 0
                shaderTree
                    {
                    }
                }
            }
        }
    }
}
'''


math_node_template = '''node math_functions @name
    {
    name @name
    pos @x @y
    advancedInputsCollapsed 0
    output Color
        {
        exposedAs @outid
        }
    nodeInput Math_Argument
        {
        name Math_Argument
        enumValue @opcode
        }
    nodeInput Value_1
        {
        name Value_1
        value 1 0 100
        exposedAs @inid1
        node NO_NODE
        }
    nodeInput Value_2
        {
        name Value_2
        value 0 0 100
        exposedAs @inid2
        node NO_NODE
        }
    }
'''


var_node_template = '''node @type @name
    {
    name @name
    pos @x @y
    advancedInputsCollapsed 0
    output Color
        {
        exposedAs @outid
        }
    }
'''


if __name__ == "__main__":
    from PIL import Image

    u = U()
    v = V()
    a = ((u - 0.5)**2 + (v - 0.5)**2).sqrt() < 0.5

    for node in a.nodes():
        print(node.format())

    out = a.data

    Image.fromarray(out * 256).show()

    name = "texgen_test"

    with open("%s.mt5" % name, "w") as fp:
        write_poser_file(fp, name, [u, v], [a])
