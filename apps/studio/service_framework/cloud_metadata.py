# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import socket
from abc import ABC, abstractmethod


class CloudMetadataProvider(ABC):
    """Abstract interface for cloud provider metadata services."""
    @abstractmethod
    def get_instance_id(self) -> str:
        """Get the current instance/node ID."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this cloud provider is available/active."""


class AWSMetadataProvider(CloudMetadataProvider):
    """AWS-specific metadata provider using EC2 metadata service."""

    # Timeout for IMDS HTTP requests. On bare-metal instances (e.g.
    # g4dn.metal) with IMDS hop limit of 1, IMDSv2 responses are
    # dropped and the socket read hangs indefinitely without a timeout.
    # Ref: https://github.com/kumo-ai/kumo/issues/27808
    _IMDS_TIMEOUT_SECONDS = 10

    def __init__(self):
        self._ec2_metadata = None
        self._is_available = None  # Cache availability check
        try:
            from ec2_metadata import ec2_metadata
            self._ec2_metadata = ec2_metadata
        except ImportError:
            pass

    def _call_imds(self):
        """Call IMDS with a socket timeout to prevent indefinite hang."""
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._IMDS_TIMEOUT_SECONDS)
        try:
            return self._ec2_metadata.instance_id
        finally:
            socket.setdefaulttimeout(old_timeout)

    def get_instance_id(self) -> str:
        """Get EC2 instance ID or fallback to hostname."""
        if self._ec2_metadata:
            try:
                return self._call_imds()
            except Exception:
                pass
        return socket.gethostname()

    def is_available(self) -> bool:
        """Check if AWS metadata service is available."""
        if self._is_available is not None:
            return self._is_available

        self._is_available = False
        if self._ec2_metadata:
            try:
                _ = self._call_imds()
                self._is_available = True
            except Exception:
                pass

        return self._is_available


class GCPMetadataProvider(CloudMetadataProvider):
    """GCP-specific metadata provider (minimal placeholder)."""
    def get_instance_id(self) -> str:
        """Get GCP instance ID or fallback to hostname."""
        # TODO: Implement when GCP support is needed
        return socket.gethostname()

    def is_available(self) -> bool:
        """Check if GCP metadata service is available."""
        # TODO: Implement when GCP support is needed
        return False


class AzureMetadataProvider(CloudMetadataProvider):
    """Azure-specific metadata provider (minimal placeholder)."""
    def get_instance_id(self) -> str:
        """Get Azure instance ID or fallback to hostname."""
        # TODO: Implement when Azure support is needed
        return socket.gethostname()

    def is_available(self) -> bool:
        """Check if Azure metadata service is available."""
        # TODO: Implement when Azure support is needed
        return False


class LocalMetadataProvider(CloudMetadataProvider):
    """Fallback metadata provider for local/development environments."""
    def get_instance_id(self) -> str:
        """Return hostname as instance ID."""
        return socket.gethostname()

    def is_available(self) -> bool:
        """Always available as fallback."""
        return True


class CloudMetadataFactory:
    """Factory for creating appropriate cloud metadata provider."""

    _PROVIDERS = [
        AWSMetadataProvider,
        GCPMetadataProvider,
        AzureMetadataProvider,
    ]

    @classmethod
    def create_provider(cls) -> CloudMetadataProvider:
        """Create appropriate metadata provider based on environment."""
        logger = logging.getLogger(__name__)

        for provider_cls in cls._PROVIDERS:
            provider = provider_cls()
            provider_name = provider_cls.__name__
            if provider.is_available():
                logger.info(f"Using cloud metadata provider: {provider_name}")
                return provider
            else:
                logger.debug(
                    f"Cloud metadata provider {provider_name} not available")

        # Fallback to local provider
        logger.info(
            "Using fallback LocalMetadataProvider (no cloud service detected)")
        return LocalMetadataProvider()
