# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and local resource layout for one devserver."""

import hashlib
import os
import os.path as osp
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional

from service_framework.deploy_env import MACHINE_INSTANCE_ID

DEFAULT_BASE_PORT = 10000
_CHILD_ID_RE = re.compile(r'^[a-z][a-z0-9-]{0,47}$')


@dataclass(frozen=True)
class DevserverInstance:
    """Resources for one devserver.

    ``base_port`` is only an address; ``child_id`` is the persistent
    local/Temporal identity, so a re-allocated port doesn't change identity.
    """

    base_port: int
    child_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.child_id is None:
            return
        normalized = unicodedata.normalize('NFKC', self.child_id).casefold()
        if not _CHILD_ID_RE.fullmatch(normalized):
            raise ValueError(
                'child_id must normalize to ASCII lowercase '
                '^[a-z][a-z0-9-]{0,47}$')
        object.__setattr__(self, 'child_id', normalized)

    @property
    def is_legacy(self) -> bool:
        return self.base_port == DEFAULT_BASE_PORT and self.child_id is None

    @property
    def name(self) -> str:
        if self.is_legacy:
            return 'devserver'
        return f'devserver-{self.child_id or self.base_port}'

    @property
    def suffix(self) -> str:
        return '' if self.is_legacy else f'/{self.name}'

    @property
    def docker_label(self) -> str:
        return f'devserver_id={self.base_port}'

    def ui_node_modules_volume(
            self, env: Mapping[str, str] = os.environ) -> Optional[str]:
        """Return the isolated Docker dependency volume for a child UI."""
        if self.is_legacy:
            return None
        home = self.environment(env)['KUMO_HOME']
        identity = hashlib.sha256(
            osp.abspath(home).encode('utf-8')).hexdigest()[:16]
        return f'kumo-ui-node-modules-{self.name}-{identity}'

    @property
    def temporal_instance_id(self) -> str:
        return self._temporal_instance_id(os.environ)

    def _temporal_instance_id(self, env: Mapping[str, str]) -> str:
        """Derive the raw child identity before DeployEnvConfig adds username."""
        if self.child_id is None:
            return self.name
        parent_identity = env.get('INSTANCE_ID') or MACHINE_INSTANCE_ID
        child_suffix = '-' + self.child_id
        return (parent_identity if parent_identity.endswith(child_suffix) else
                parent_identity + child_suffix)

    def data_root(self, root: str) -> str:
        return root if self.is_legacy else root.rstrip('/') + self.suffix

    def port(self, offset: int) -> int:
        return self.base_port + offset

    def ports(self) -> Dict[str, int]:
        """The complete port allocation, with fixed legacy exceptions."""
        ports = {
            'web_ui': self.port(0),
            'rest_api': self.port(1),
            'public_rest_api': self.port(2),
            'grafana': self.port(4),
            'pool_worker_metrics': self.port(5),
            'pool_metrics': self.port(6),
            'pool_worker_worker_metrics': self.port(7),
            'rest_worker_metrics': self.port(8),
            'rest_metrics': self.port(9),
            'data_worker_metrics': self.port(10),
            'ml_worker_metrics': self.port(11),
            'public_rest_worker_metrics': self.port(12),
            'public_rest_metrics': self.port(13),
        }
        if self.is_legacy:
            ports.update(jaeger_ui=16686, jaeger_otlp=4317, prometheus=9090)
        else:
            ports.update(jaeger_ui=self.port(14), jaeger_otlp=self.port(15),
                         prometheus=self.port(16))
        return ports

    def environment(self, env: Mapping[str,
                                       str] = os.environ) -> Dict[str, str]:
        """Return launcher environment overrides for this instance.

        The default instance intentionally returns no overrides so its
        long-standing paths and environment are unchanged.
        """
        if self.is_legacy:
            return {}
        base_home = env.get(
            'KUMO_HOME', osp.join(os.getcwd(), '.cache', 'kumo', 'devserver'))
        # ``web_devserver`` may be called after the CLI has already applied
        # this instance.  Do not turn ``.../devserver-11000`` into
        # ``.../devserver-11000/devserver-11000`` in that normal path.
        home = (base_home if osp.basename(osp.normpath(base_home)) == self.name
                else osp.join(base_home, self.name))
        # Each child gets isolated XDG roots, but mise-managed executables must
        # continue using the operator's installed tool data and cache. If mise
        # follows the new empty XDG roots, invoking one of its shims can trigger
        # recursive tool-version resolution (including npm invoking its own
        # mise shim) before the devserver binds any listener.
        operator_home = env.get('HOME', osp.expanduser('~'))
        operator_xdg_cache = env.get('XDG_CACHE_HOME',
                                     osp.join(operator_home, '.cache'))
        operator_xdg_data = env.get(
            'XDG_DATA_HOME', osp.join(operator_home, '.local', 'share'))
        mise_cache_dir = env.get('MISE_CACHE_DIR',
                                 osp.join(operator_xdg_cache, 'mise'))
        mise_data_dir = env.get('MISE_DATA_DIR',
                                osp.join(operator_xdg_data, 'mise'))
        # Keep TMPDIR short. PyTorch's ``file_system`` sharing strategy adds
        # ``torch-shm-dir-XXXXXXXX/manager.sock`` below this path, and AF_UNIX
        # socket paths are limited to roughly 108 bytes on Linux. Nesting the
        # temp root below an arbitrary KUMO_HOME makes otherwise valid local
        # training fail when torch_shm_manager binds its socket. The hash keeps
        # concurrent child instances isolated without exposing the full home
        # path to that fixed kernel limit.
        tmp_identity = hashlib.sha256(
            osp.abspath(home).encode('utf-8')).hexdigest()[:12]
        tmpdir = osp.join('/tmp', f'kumo-dev-{tmp_identity}')
        return {
            'DB_STORAGE_ROOT': osp.join(home, 'db'),
            'DEVSERVER_INSTANCE_SUFFIX': self.suffix,
            'HF_HOME': osp.join(home, 'hf'),
            'INSTANCE_ID': self._temporal_instance_id(env),
            'KUMO_AUTORPC_SERVER_NAME_PREFIX': self.name,
            'KUMO_GRAFANA_HOME': osp.join(home, 'monitoring'),
            'KUMO_HOME': home,
            'MISE_CACHE_DIR': mise_cache_dir,
            'MISE_DATA_DIR': mise_data_dir,
            'TMPDIR': tmpdir,
            'TORCH_HOME': osp.join(home, 'torch'),
            'XDG_CACHE_HOME': osp.join(home, 'xdg-cache'),
            'XDG_CONFIG_HOME': osp.join(home, 'xdg-config'),
            'XDG_DATA_HOME': osp.join(home, 'xdg-data'),
        }

    def apply_environment(self, env: MutableMapping[str,
                                                    str] = os.environ) -> None:
        overrides = self.environment(env)
        self.prepare_local_paths(overrides)
        env.update(overrides)

    def prepare_local_paths(self, env: Mapping[str, str]) -> None:
        """Create child-private roots before launch.

        ``TMPDIR`` deliberately uses a short, hashed path under ``/tmp`` so
        PyTorch shared-memory manager sockets stay below the AF_UNIX path
        limit while remaining isolated between child instances.
        """
        if self.is_legacy:
            return
        for name in (
                'KUMO_HOME', 'DB_STORAGE_ROOT', 'HF_HOME',
                'KUMO_GRAFANA_HOME', 'TMPDIR', 'TORCH_HOME',
                'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME'):
            path = env[name]
            os.makedirs(path, mode=0o700, exist_ok=True)
