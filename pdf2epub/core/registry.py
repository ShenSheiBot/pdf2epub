"""
Component registry: centralized registration of processors and validators.

All components must be registered here. Direct instantiation of unregistered
components is discouraged (use get_* methods instead).

FROZEN: This class cannot be inherited or modified.
"""

from typing import Dict, Type, TypeVar, Callable, Any, List, Union
from loguru import logger

from ._frozen import Frozen, final, check_final_methods
from ._protocol import ProcessorProtocol, IndividualValidator, BatchValidator


T = TypeVar('T')


@check_final_methods
class ComponentRegistry(Frozen, frozen=True):
    """
    Central registry for all components.

    FROZEN: Cannot be inherited.

    Usage:
        # Register
        ComponentRegistry.register_processor("polish", PolishProcessor)
        ComponentRegistry.register_validator("ngram", NGramValidator)

        # Get
        processor = ComponentRegistry.get_processor("polish", **kwargs)
        validator = ComponentRegistry.get_validator("ngram", **kwargs)

        # List
        names = ComponentRegistry.list_processors()
    """

    _processors: Dict[str, Type] = {}
    _validators: Dict[str, Type] = {}
    _processor_factories: Dict[str, Callable] = {}
    _validator_factories: Dict[str, Callable] = {}

    # ========== PROCESSOR REGISTRATION ==========

    @classmethod
    @final
    def register_processor(
        cls,
        name: str,
        processor_cls: Type,
        factory: Callable = None
    ) -> None:
        """
        Register a processor.

        Args:
            name: Unique name for the processor
            processor_cls: Processor class (must implement ProcessorProtocol)
            factory: Optional factory function for creating instances

        Raises:
            ValueError: If name already registered
            TypeError: If class doesn't implement ProcessorProtocol
        """
        if name in cls._processors:
            raise ValueError(
                f"\n{'='*60}\n"
                f"REGISTRATION ERROR: Processor '{name}' already registered\n"
                f"{'='*60}\n"
                f"Each processor must have a unique name.\n"
                f"If you need a variant, use a different name.\n"
                f"{'='*60}"
            )

        # Verify protocol compliance (basic check)
        required_methods = ['build_prompt', 'clean_response', 'post_process', 'get_model_configs']
        for method in required_methods:
            if not hasattr(processor_cls, method):
                raise TypeError(
                    f"Processor '{name}' ({processor_cls.__name__}) missing required method '{method}'"
                )

        cls._processors[name] = processor_cls
        if factory:
            cls._processor_factories[name] = factory

        logger.debug(f"Registered processor: {name}")

    @classmethod
    @final
    def get_processor(cls, name: str, **kwargs) -> ProcessorProtocol:
        """
        Get a processor instance.

        Args:
            name: Processor name
            **kwargs: Arguments to pass to constructor/factory

        Returns:
            Processor instance

        Raises:
            KeyError: If processor not registered
        """
        if name not in cls._processors:
            available = list(cls._processors.keys())
            raise KeyError(
                f"\n{'='*60}\n"
                f"UNKNOWN PROCESSOR: '{name}'\n"
                f"{'='*60}\n"
                f"Available processors: {available}\n"
                f"\n"
                f"Make sure the processor module has been imported.\n"
                f"{'='*60}"
            )

        if name in cls._processor_factories:
            return cls._processor_factories[name](**kwargs)
        return cls._processors[name](**kwargs)

    @classmethod
    @final
    def list_processors(cls) -> List[str]:
        """List all registered processor names."""
        return list(cls._processors.keys())

    # ========== VALIDATOR REGISTRATION ==========

    @classmethod
    @final
    def register_validator(
        cls,
        name: str,
        validator_cls: Type,
        factory: Callable = None
    ) -> None:
        """
        Register a validator.

        Args:
            name: Unique name for the validator
            validator_cls: Validator class (must implement IndividualValidator or BatchValidator)
            factory: Optional factory function for creating instances

        Raises:
            ValueError: If name already registered
        """
        if name in cls._validators:
            raise ValueError(
                f"\n{'='*60}\n"
                f"REGISTRATION ERROR: Validator '{name}' already registered\n"
                f"{'='*60}\n"
                f"Each validator must have a unique name.\n"
                f"{'='*60}"
            )

        # Verify protocol compliance (basic check)
        # Individual validators have 'validate', Batch validators have 'validate_batch'
        has_validate = hasattr(validator_cls, 'validate')
        has_validate_batch = hasattr(validator_cls, 'validate_batch')

        if not has_validate and not has_validate_batch:
            raise TypeError(
                f"Validator '{name}' ({validator_cls.__name__}) must implement "
                f"either 'validate' (IndividualValidator) or 'validate_batch' (BatchValidator)"
            )

        if not hasattr(validator_cls, 'name'):
            raise TypeError(
                f"Validator '{name}' ({validator_cls.__name__}) missing required 'name' property"
            )

        cls._validators[name] = validator_cls
        if factory:
            cls._validator_factories[name] = factory

        logger.debug(f"Registered validator: {name}")

    @classmethod
    @final
    def get_validator(cls, name: str, **kwargs) -> Union[IndividualValidator, BatchValidator]:
        """
        Get a validator instance.

        Args:
            name: Validator name
            **kwargs: Arguments to pass to constructor/factory

        Returns:
            Validator instance (IndividualValidator or BatchValidator)

        Raises:
            KeyError: If validator not registered
        """
        if name not in cls._validators:
            available = list(cls._validators.keys())
            raise KeyError(
                f"\n{'='*60}\n"
                f"UNKNOWN VALIDATOR: '{name}'\n"
                f"{'='*60}\n"
                f"Available validators: {available}\n"
                f"{'='*60}"
            )

        if name in cls._validator_factories:
            return cls._validator_factories[name](**kwargs)
        return cls._validators[name](**kwargs)

    @classmethod
    @final
    def list_validators(cls) -> List[str]:
        """List all registered validator names."""
        return list(cls._validators.keys())

    # ========== UTILITIES ==========

    @classmethod
    @final
    def get_individual_validators(cls, **kwargs) -> List[IndividualValidator]:
        """
        Get all individual validators (those with 'validate' method).

        Args:
            **kwargs: Arguments to pass to validators

        Returns:
            List of individual validator instances
        """
        result = []
        for name in cls._validators:
            validator = cls.get_validator(name, **kwargs)
            if hasattr(validator, 'validate') and callable(getattr(validator, 'validate')):
                result.append(validator)
        return result

    @classmethod
    @final
    def get_batch_validators(cls, **kwargs) -> List[BatchValidator]:
        """
        Get all batch validators (those with 'validate_batch' method).

        Args:
            **kwargs: Arguments to pass to validators

        Returns:
            List of batch validator instances
        """
        result = []
        for name in cls._validators:
            validator = cls.get_validator(name, **kwargs)
            if hasattr(validator, 'validate_batch') and callable(getattr(validator, 'validate_batch')):
                result.append(validator)
        return result

    @classmethod
    @final
    def clear_all(cls) -> None:
        """
        Clear all registrations.

        WARNING: Only use in tests!
        """
        cls._processors.clear()
        cls._validators.clear()
        cls._processor_factories.clear()
        cls._validator_factories.clear()
        logger.warning("ComponentRegistry cleared - all registrations removed")


# ========== DECORATOR FOR EASY REGISTRATION ==========

def register_processor(name: str):
    """
    Decorator to register a processor class.

    Usage:
        @register_processor("polish")
        class PolishProcessor:
            ...
    """
    def decorator(cls):
        ComponentRegistry.register_processor(name, cls)
        return cls
    return decorator


def register_validator(name: str):
    """
    Decorator to register a validator class.

    Usage:
        @register_validator("ngram")
        class NGramValidator:
            ...
    """
    def decorator(cls):
        ComponentRegistry.register_validator(name, cls)
        return cls
    return decorator
