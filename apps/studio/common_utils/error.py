# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type

from temporalio.exceptions import TemporalError


class PublicMessage:
    """
    Base class for all user-facing messages (errors, warnings, info).
    Supports string templating via `template`
    """
    template: str = ""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        try:
            self.message = self.template.format(**kwargs)
        except KeyError as e:
            self.message = f"{self.template} [Missing key: {e.args[0]}]"

    def __str__(self):
        return self.message


@dataclass(init=False)
class ErrorInfo:
    """
    Structured error information for user-facing error messages.

    Attributes:
        source: The system/service that raised the error (e.g., "Databricks",
            "Spark service", "Snowflake").
        exception_msg: The error message to display. Can be either the raw
            exception string or a summarized/parsed version.
        friendly: Optional additional friendly explanation or hint for the
            user about what might have gone wrong or how to resolve it.
        is_raw: True if exception_msg is the raw str(err); False if it was
            summarized or interpreted by a friendly_fn.
        stop_chain: True when the classified exception is a trust boundary and
            its nested cause must not contribute to the public message.
    """
    source: Optional[str]
    exception_msg: Optional[str]
    friendly: Optional[str] = None
    is_raw: bool = False
    stop_chain: bool = False

    def __init__(self, source: Optional[str],
                 exception_msg: Optional[str] = None,
                 friendly: Optional[str] = None, is_raw: bool = False,
                 stop_chain: bool = False):
        self.source = source
        self.exception_msg = exception_msg
        self.friendly = friendly
        self.is_raw = is_raw
        self.stop_chain = stop_chain


Classifier = Tuple[str, Type[Exception], Optional[Callable[[Exception],
                                                           ErrorInfo]]]


class PublicException(PublicMessage, Exception):
    """
    Base user-facing error that builds a friendly message from an optional
    exception.

    Classification flow:
      - CLASSIFIERS define how to handle different exception types.
      - A classifier returns an ErrorInfo (source, exception_msg, friendly).
        exception_msg may be None for friendly-only cases.
      - Subclasses define their own CLASSIFIERS list, which are combined
        with parent classifiers via MRO (most specific first).
      - If you pass `error`, this class appends messages from the error chain
        to `message`. Avoid `message=str(error)` with `error=error`, or the
        error text will appear twice.

    Subclasses define CLASSIFIERS; no need to override __init__.
    """
    non_retryable: bool = True
    http_code: int = 422
    template: str = "{message}"
    CLASSIFIERS: List[Classifier] = []

    def __init__(self, message: str = "", error: Optional[Exception] = None,
                 related_errors: Optional[List[Exception]] = None,
                 non_retryable: Optional[bool] = None,
                 suffix_unclassified_error=False, **kwargs):
        if not message and 'message' in kwargs:
            message = kwargs.pop('message')

        # Walk exception chain and collect messages
        if error is not None:
            collected_messages: List[str] = [message] if message else []

            # Walk the __cause__ chain starting from error
            # Cap at 20 levels to prevent infinite loops
            current = error
            depth = 0
            max_depth = 20
            while current is not None and depth < max_depth:
                if isinstance(current, PublicException):
                    # Extract PublicException message and STOP
                    # (it has already processed its own chain previously)
                    collected_messages.append(current.message)
                    break

                # Skip builtin Python exceptions (ValueError, TypeError, etc.)
                # and Temporal framework exceptions (ActivityError, etc.),
                # and continue walking to find useful errors deeper in chain
                if (type(current).__module__ == 'builtins'
                        or isinstance(current, TemporalError)):
                    current = getattr(current, '__cause__', None)
                    continue

                # Process non-builtin, non-PublicException errors
                info = self._error_info(current)

                if info.exception_msg:
                    # Format the message based on classifier result
                    if info.source:
                        # Classified error (e.g., Snowflake, Databricks)
                        if info.is_raw:
                            collected_messages.append(
                                f"{info.source} returned an error:")
                        else:
                            collected_messages.append(f"{info.source} error:")
                        collected_messages.append(info.exception_msg)
                    else:
                        # Custom/library exception without classifier
                        collected_messages.append(
                            "The following error occurred:")
                        collected_messages.append(f"{type(current).__name__}: "
                                                  f"{info.exception_msg}")

                if info.friendly:
                    collected_messages.append(info.friendly)

                if info.stop_chain:
                    break

                # Continue to next exception in chain
                current = getattr(current, '__cause__', None)
                depth += 1

            is_unclassified = bool(message) and len(collected_messages) == 1

            if is_unclassified and suffix_unclassified_error:
                # The error could not be classified, so we append a friendlier message
                message = (f"{message.rstrip(' .')}. Please contact "
                           "Kumo to investigate and resolve the issue.")
            else:
                message = "\n".join(collected_messages)

        # Store message in kwargs for template formatting
        if message:
            kwargs = {'message': message}
        super().__init__(**kwargs)
        self.related_errors = related_errors or []
        if non_retryable is not None:
            self.non_retryable = non_retryable

    def get_user_message(self, indent_level: int = 0) -> str:
        """
        Get the user-facing error message.

        Includes messages from any related PublicExceptions automatically.
        Standard exceptions in related_errors are excluded (internal only).

        Args:
            indent_level: Internal parameter for recursive indentation
        """
        indent = "  " * indent_level
        if indent_level > 0:
            message = f"{indent}{self.message}"
        else:
            message = self.message

        if not self.related_errors:
            return message

        # Only include PublicException related errors
        related_indent = "  " * (indent_level + 1)
        related_messages = [
            e.get_user_message(indent_level + 1) for e in self.related_errors
            if isinstance(e, PublicException)
        ]

        if not related_messages:
            return message

        if indent_level == 0:
            additional = "\n\nRelated errors:\n"
        else:
            additional = "\n"
        formatted_msgs = "\n".join(f"{related_indent}- {msg.lstrip()}"
                                   for msg in related_messages)
        return f"{message}{additional}{formatted_msgs}"

    def add_related_error(self, error: Exception):
        """
        Add an exception that provides additional context to the main failure.

        Common uses:
        - Individual failures from parallel operations
        - Cleanup/rollback failures after the main error
        - Cascading failures triggered by the primary failure
        - Related warnings for additional context

        Related errors are logged for debugging but don't affect retry logic
        or HTTP status codes.

        Args:
            error: Any exception (PublicException or standard Exception)
        """
        self.related_errors.append(error)

    def to_dict(self) -> dict:
        """Serialize exception data for Temporal ApplicationError details"""
        return {
            'type':
            self.__class__.__name__,
            'message':
            self.message,
            'template':
            self.template,
            'kwargs':
            self.kwargs,
            'http_code':
            self.http_code,
            'non_retryable':
            self.non_retryable,
            'related_errors':
            [
                e.to_dict() for e in self.related_errors
                if isinstance(e, PublicException)
            ]
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct PublicException from error data"""
        # Find the correct subclass by type name
        exception_class = cls._get_exception_class(data['type'])

        # Reconstruct related errors
        related_errors = []
        for error_data in data.get('related_errors', []):
            # Duck typing: if it has 'template' field, it's a PublicException
            if 'template' in error_data:
                related_errors.append(cls.from_dict(error_data))
            else:
                # Standard exception - try to recreate with correct type
                exc_class = cls._get_standard_exception_class(
                    error_data['type'], error_data.get('module'))
                related_errors.append(exc_class(error_data['message']))

        return exception_class(related_errors=related_errors, **data['kwargs'])

    @classmethod
    def _get_exception_class(cls, type_name: str):
        """Find exception subclass by name (recursive search)"""
        def find_subclass(base_class):
            if base_class.__name__ == type_name:
                return base_class
            for subclass in base_class.__subclasses__():
                result = find_subclass(subclass)
                if result:
                    return result
            return None

        result = find_subclass(cls)
        return result if result else cls  # fallback to base class

    @staticmethod
    def _get_standard_exception_class(type_name: str,
                                      module: Optional[str] = None):
        """Get standard exception class by name and module"""
        # Try builtins first (most common)
        if module == 'builtins' or module is None:
            try:
                # __builtins__ can be either a dict or a module depending on
                # context
                if isinstance(__builtins__, dict):
                    exc_class = __builtins__.get(type_name)
                    if exc_class:
                        return exc_class
                else:
                    exc_class = getattr(__builtins__, type_name, None)
                    if exc_class:
                        return exc_class
            except (AttributeError, TypeError):
                pass

        # For third-party exceptions (psycopg2.Error,
        # requests.RequestException, etc.) we can't reliably reconstruct them
        # without importing the module. Fall back to generic Exception
        return Exception

    @classmethod
    def _all_classifiers(cls):
        """
        Collect all classifiers from the class hierarchy using MRO.
        Most specific classes come first (checked first for matches).
        Only includes classes that actually define CLASSIFIERS (not inherited).
        """
        all_classifiers = []
        # Traverse MRO forward: child classifiers take precedence over parent
        for base_cls in cls.__mro__:
            # Only include if the class actually defines CLASSIFIERS
            # (not just inherits it)
            if 'CLASSIFIERS' in base_cls.__dict__:
                all_classifiers.extend(base_cls.CLASSIFIERS)
        return all_classifiers

    @classmethod
    def _error_info(cls, err: Exception) -> ErrorInfo:
        """
        Pick the first matching classifier and return a consolidated ErrorInfo.
        - source: classifier source or override from friendly_fn
        - exception_msg: friendly_fn.exception_msg if set; otherwise str(err)
        - friendly: optional friendly text from friendly_fn
        - is_raw: True if exception_msg is raw str(err), False if summarized

        Note: Builtin Python exceptions (ValueError, TypeError, etc.) and
        Temporal framework exceptions (ActivityError, ApplicationError, etc.)
        are filtered out before this method is called, so we only receive
        custom/library exceptions or classified errors here.
        """
        for source, exc_type, friendly_fn in cls._all_classifiers():
            if not isinstance(err, exc_type):
                continue
            info = friendly_fn(err) if friendly_fn else ErrorInfo(
                source=None, exception_msg=None, friendly=None)
            info_source = info.source or source
            # Respect a friendly_fn that intentionally leaves exception_msg as
            # None; only default to str(err) when no friendly_fn is provided.
            if friendly_fn:
                info_exception_msg = info.exception_msg
                is_raw = False
            else:
                info_exception_msg = str(err)
                is_raw = True
            return ErrorInfo(source=info_source,
                             exception_msg=info_exception_msg,
                             friendly=info.friendly, is_raw=is_raw,
                             stop_chain=info.stop_chain)

        # No classifier matched - this is a custom/library exception
        # Expose with raw message (builtins already filtered out)
        return ErrorInfo(source=None, exception_msg=str(err), friendly=None,
                         is_raw=True)
