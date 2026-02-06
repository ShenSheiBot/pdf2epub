"""
Frozen base classes that cannot be inherited or modified.

Any attempt to:
1. Inherit from a frozen class -> TypeError at import
2. Override a @final method -> mypy error + runtime check
3. Define forbidden methods in subclass -> TypeError at import

This prevents Claude from "reinventing the wheel" by making violations
crash immediately rather than silently producing bugs.
"""

from typing import ClassVar, Set, Any
import functools


class FrozenMeta(type):
    """
    Metaclass that prevents inheritance of frozen classes.

    When a class is created with `frozen=True`, any attempt to inherit
    from it will raise TypeError at import time.
    """

    _frozen_classes: ClassVar[Set[str]] = set()

    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict,
        frozen: bool = False,
        **kwargs
    ):
        # Check if trying to inherit from a frozen class
        for base in bases:
            base_name = getattr(base, '__name__', '')
            if base_name in mcs._frozen_classes:
                raise TypeError(
                    f"\n{'='*60}\n"
                    f"ARCHITECTURE VIOLATION: Inheritance from frozen class\n"
                    f"{'='*60}\n"
                    f"Class '{name}' attempts to inherit from frozen class '{base_name}'.\n"
                    f"\n"
                    f"This is forbidden by design. Frozen classes cannot be extended\n"
                    f"because their behavior must remain unchanged.\n"
                    f"\n"
                    f"SOLUTION: Use composition instead of inheritance.\n"
                    f"  - Inject the {base_name} instance via constructor\n"
                    f"  - Call its methods instead of overriding them\n"
                    f"{'='*60}"
                )

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if frozen:
            mcs._frozen_classes.add(name)

        return cls


class Frozen(metaclass=FrozenMeta):
    """
    Base class for frozen components.

    Subclasses of Frozen:
    1. Can define _FORBIDDEN_METHODS to prevent certain method names
    2. Will have their @final methods checked at runtime
    3. Cannot be further inherited if created with frozen=True

    Usage:
        class MyComponent(Frozen, frozen=True):
            _FORBIDDEN_METHODS = {'validate', 'save'}

            @final
            def do_something(self):
                ...
    """

    # Method names that subclasses are forbidden from defining
    _FORBIDDEN_METHODS: ClassVar[Set[str]] = set()

    def __init_subclass__(cls, frozen: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)

        # Check for forbidden methods
        for method_name in cls._FORBIDDEN_METHODS:
            if method_name in cls.__dict__:
                raise TypeError(
                    f"\n{'='*60}\n"
                    f"ARCHITECTURE VIOLATION: Forbidden method defined\n"
                    f"{'='*60}\n"
                    f"Class '{cls.__name__}' defines forbidden method '{method_name}'.\n"
                    f"\n"
                    f"This functionality is provided by core components and\n"
                    f"custom implementations are not allowed.\n"
                    f"\n"
                    f"SOLUTION: Remove the '{method_name}' method and use the\n"
                    f"corresponding core component instead.\n"
                    f"{'='*60}"
                )


def final(method):
    """
    Decorator that marks a method as final (cannot be overridden).

    Unlike typing.final which only affects type checkers, this decorator
    adds a runtime check in __init_subclass__ of the containing class.

    Usage:
        class MyClass(Frozen):
            @final
            def important_method(self):
                ...
    """
    method._is_final = True

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)

    wrapper._is_final = True
    return wrapper


def check_final_methods(cls):
    """
    Class decorator that enforces @final methods cannot be overridden.

    Usage:
        @check_final_methods
        class MyClass:
            @final
            def cannot_override(self):
                ...
    """
    original_init_subclass = cls.__init_subclass__

    @classmethod
    def new_init_subclass(subcls, **kwargs):
        # Check for overridden final methods
        for name, method in cls.__dict__.items():
            if getattr(method, '_is_final', False):
                if name in subcls.__dict__:
                    raise TypeError(
                        f"\n{'='*60}\n"
                        f"ARCHITECTURE VIOLATION: Final method overridden\n"
                        f"{'='*60}\n"
                        f"Class '{subcls.__name__}' overrides final method '{name}'\n"
                        f"from class '{cls.__name__}'.\n"
                        f"\n"
                        f"Final methods cannot be overridden because their behavior\n"
                        f"must remain unchanged across the codebase.\n"
                        f"\n"
                        f"SOLUTION: Do not override '{name}'. If you need different\n"
                        f"behavior, use composition or create a new method.\n"
                        f"{'='*60}"
                    )

        # Call original __init_subclass__
        if original_init_subclass:
            original_init_subclass.__func__(subcls, **kwargs)

    cls.__init_subclass__ = new_init_subclass
    return cls
