class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._children = set(_children)
        self._backward = lambda: None

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad -= 1 * out.grad

        out._backward = _backward
        return out

    def log(self):
        from math import log

        out = Value(log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        topo.reverse()

        self.grad = 1.0
        for node in topo:
            node._backward()


x = Value(2.0)
loss = x * x.log()
loss.backward()

print(loss.data)
print(loss.grad)
print(x.grad)
