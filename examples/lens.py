from confingy import Lazy, lens, track


@track
class Value:
    def __init__(self, value: int):
        self.value = value


@track
class Container:
    def __init__(self, values: list[Value]):
        self.values = values


@track
class Wrapper:
    def __init__(self, container: Lazy[Container]):
        self.container = container


# Wrapper contains both lazy and non-lazy components.
wrapper = Wrapper(container=Container.lazy(values=[Value(value=42), Value(value=666)]))
# We create a lens on the wrapper which allows us to manipulate any components
# and their constructor arguments.
wrapper_lens = lens(wrapper)
print(wrapper_lens.container.values[0].value)  # 42
# We can modify values via "dot" access.
wrapper_lens.container.values[0].value = 1000
print(wrapper_lens.container.values[0].value)  # 1000
# In order to get back the original wrapper object, we "unlens" it.
wrapper = wrapper_lens.unlens()
print(wrapper.container.instantiate().values[0].value)
