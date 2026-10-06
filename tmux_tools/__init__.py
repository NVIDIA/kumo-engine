# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Devserver

This package allows the user to configure developer environments consisting of
multiple TMUX windows running in parallel, which may expose one or more HTTP
ports that require SSH tunneling from the user's laptop.

The following features are provided:

- Automatically spin up multiple tmux windows, each running a different command
- Automatic generation of the SSH port-forwarding command for AWS instances
- Ability to reconnect to an already-running devserver
- Dependency checking, to notify the user of missing binaries or packages
- A default tmux.conf, to make it easy for those unfamiliar with tmux
- Declarative config-based syntax for configuring TMUX windows


In order to run the devserver, you can run this command from the root of the
kumo repository:

.. code-block:: bash

   python3 rest/devserver/main.py <config_name>

If you want to add a new configuration, you can directly edit main.py.

By default, the devserver uses a base port of 10000. This automatically enables
live authentication.

If you'd like to use a random port and disable authentication, you can pass
an additional --random port flag to the command as follows:

.. code-block:: bash

   python3 rest/devserver/main.py <config_name> --random_port
"""
