"""Tests for confingy.hashing module - class bytecode hashing functionality."""

from confingy.utils.hashing import hash_class


class TestHashClass:
    """Tests for the hash_class function."""

    def test_basic_class_hash(self):
        """Test that hash_class produces consistent hashes for classes."""

        class SimpleClass:
            def __init__(self, value: int):
                self.value = value

        hash1 = hash_class(SimpleClass)
        hash2 = hash_class(SimpleClass)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_different_classes_different_hashes(self):
        """Test that different classes have different hashes."""

        class ClassA:
            def method(self):
                return 1

        class ClassB:
            def method(self):
                return 2

        hash_a = hash_class(ClassA)
        hash_b = hash_class(ClassB)

        assert hash_a != hash_b

    def test_class_with_same_structure_different_name(self):
        """Test that classes with same structure but different names have different hashes."""

        class FirstClass:
            def compute(self, x):
                return x * 2

        class SecondClass:
            def compute(self, x):
                return x * 2

        # Different class names should result in different hashes
        assert hash_class(FirstClass) != hash_class(SecondClass)

    def test_method_implementation_changes_hash(self):
        """Test that changing method implementation changes the hash."""

        class VersionOne:
            def process(self, data):
                return data * 10

        class VersionTwo:
            def process(self, data):
                return data * 20  # Different implementation

        assert hash_class(VersionOne) != hash_class(VersionTwo)

    def test_added_method_changes_hash(self):
        """Test that adding a method changes the hash."""

        class Basic:
            def __init__(self):
                pass

        class WithMethod:
            def __init__(self):
                pass

            def new_method(self):
                return "hello"

        assert hash_class(Basic) != hash_class(WithMethod)

    def test_class_variables_affect_hash(self):
        """Test that class variables are included in the hash."""

        class WithConstantA:
            CONSTANT = 100

            def get_constant(self):
                return self.CONSTANT

        class WithConstantB:
            CONSTANT = 200  # Different value

            def get_constant(self):
                return self.CONSTANT

        assert hash_class(WithConstantA) != hash_class(WithConstantB)

    def test_property_affects_hash(self):
        """Test that properties affect the class hash."""

        class WithoutProperty:
            def __init__(self):
                self._value = 0

        class WithProperty:
            def __init__(self):
                self._value = 0

            @property
            def value(self):
                return self._value

        assert hash_class(WithoutProperty) != hash_class(WithProperty)

    def test_staticmethod_classmethod_affect_hash(self):
        """Test that static and class methods affect the hash."""

        class PlainMethod:
            def helper(self, x):
                return x * 2

        class StaticMethod:
            @staticmethod
            def helper(x):
                return x * 2

        class ClassMethod:
            @classmethod
            def helper(cls, x):
                return x * 2

        # All three should have different hashes
        plain_hash = hash_class(PlainMethod)
        static_hash = hash_class(StaticMethod)
        class_hash = hash_class(ClassMethod)

        assert plain_hash != static_hash
        assert plain_hash != class_hash
        assert static_hash != class_hash

    def test_inheritance_affects_hash(self):
        """Test that inheritance affects the class hash."""

        class Parent:
            def parent_method(self):
                return "parent"

        class Child(Parent):
            def child_method(self):
                return "child"

        # Parent and child should have different hashes
        assert hash_class(Parent) != hash_class(Child)

    def test_docstring_does_not_affect_hash(self):
        """Test that docstrings do NOT affect the hash (they don't affect execution).

        Since class names are part of the hash, we need to dynamically create classes
        with the same name but different docstrings.
        """
        import types

        # Create first class with docstrings
        def method_with_docstring(self):
            """Method docstring."""
            return 42

        def make_method_with_docstring(ns):
            return ns.update(
                {"__doc__": "Class docstring.", "method": method_with_docstring}
            )

        ClassWithDocstring = types.new_class(
            "TestClass",
            bases=(),
            kwds={},
            exec_body=make_method_with_docstring,
        )

        # Create second class without docstrings
        def method_without_docstring(self):
            return 42

        def make_method_without_docstring(ns):
            return ns.update({"__doc__": None, "method": method_without_docstring})

        ClassWithoutDocstring = types.new_class(
            "TestClass",
            bases=(),
            kwds={},
            exec_body=make_method_without_docstring,
        )

        # Both classes have the same name and structure, only differ in docstrings
        hash1 = hash_class(ClassWithDocstring)
        hash2 = hash_class(ClassWithoutDocstring)

        # They should have the same hash since docstrings don't affect execution
        assert hash1 == hash2, (
            f"Hashes differ despite only docstring differences: {hash1} != {hash2}"
        )

    def test_string_constants_affect_hash(self):
        """Test that actual string constants DO affect the hash."""

        class WithStringA:
            def method(self):
                return "hello"

        class WithStringB:
            def method(self):
                return "world"

        # Different string constants mean different functionality
        assert hash_class(WithStringA) != hash_class(WithStringB)

    def test_none_constant_preserved_when_used(self):
        """Test that None constants that are actually used in code are preserved."""

        class ReturnsNone:
            def method(self):
                return None

        class ReturnsZero:
            def method(self):
                return 0

        # These classes have different behavior, so hashes should differ
        assert hash_class(ReturnsNone) != hash_class(ReturnsZero)


class TestCompareClassBytecode:
    """Tests for the compare_class_bytecode function."""

    def test_identical_classes(self):
        """Test comparing identical classes."""

        class TestClass:
            def method(self):
                return 1

        assert hash_class(TestClass) == hash_class(TestClass)

    def test_different_classes(self):
        """Test comparing different classes."""

        class ClassOne:
            def method(self):
                return 1

        class ClassTwo:
            def method(self):
                return 2

        assert hash_class(ClassOne) != hash_class(ClassTwo)

    def test_functionally_identical_classes(self):
        """Test comparing functionally identical classes with different names."""

        class Alpha:
            def compute(self, x):
                return x + 10

        class Beta:
            def compute(self, x):
                return x + 10

        # Different names mean different hashes
        assert hash_class(Alpha) != hash_class(Beta)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_class(self):
        """Test hashing an empty class."""

        class Empty:
            pass

        hash_val = hash_class(Empty)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_class_with_init_only(self):
        """Test class with only __init__ method."""

        class InitOnly:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        hash_val = hash_class(InitOnly)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_class_with_special_methods(self):
        """Test class with special methods like __str__, __repr__."""

        class SpecialMethods:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"Value: {self.value}"

            def __repr__(self):
                return f"SpecialMethods({self.value})"

        hash_val = hash_class(SpecialMethods)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_nested_classes(self):
        """Test hashing classes with nested classes."""

        class Outer:
            class Inner:
                def inner_method(self):
                    return "inner"

            def outer_method(self):
                return "outer"

        # Should be able to hash both
        outer_hash = hash_class(Outer)
        inner_hash = hash_class(Outer.Inner)

        assert outer_hash != inner_hash
        assert isinstance(outer_hash, str)
        assert isinstance(inner_hash, str)

    def test_class_with_lambda(self):
        """Test class with lambda expressions."""

        class WithLambda:
            get_double = lambda self, x: x * 2  # noqa: E731

        hash_val = hash_class(WithLambda)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64

    def test_different_hash_algorithms(self):
        """Test using different hash algorithms."""

        class TestClass:
            def method(self):
                return 42

        sha256_hash = hash_class(TestClass, algorithm="sha256")
        md5_hash = hash_class(TestClass, algorithm="md5")

        # Different algorithms produce different length hashes
        assert len(sha256_hash) == 64  # SHA256
        assert len(md5_hash) == 32  # MD5
        assert sha256_hash != md5_hash
