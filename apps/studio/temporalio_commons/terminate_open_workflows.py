# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import getpass
import logging

import temporalio.api.workflowservice.v1 as wsv1

from service_framework.deploy_env import DeployStage
from service_framework.deploy_env_config import deploy_env
from temporalio_commons.temporal_client import TemporalClient

log = logging.getLogger(__name__)


async def run(temporal_config=None, ignore_error: bool = False):
    temporal_config = temporal_config or deploy_env()
    if temporal_config.deploy_mode == DeployStage.PROD:
        raise RuntimeError(
            "For stability reasons, you are not permitted to "
            "terminate prod workflows by default. Please edit the "
            "code and re - run the script if you truly intend to "
            "terminate the prod workflows.")

    temporal_client = TemporalClient(temporal_config)
    client = await temporal_client.get()
    namespace = temporal_config.temporal_namespace
    task_queue_suffix = temporal_config.task_queue('')

    next_page_token = b''
    while True:
        try:
            resp = await client.workflow_service.list_open_workflow_executions(
                wsv1.ListOpenWorkflowExecutionsRequest(
                    namespace=namespace, next_page_token=next_page_token),
                retry=True)
        except Exception:  # pylint: disable=broad-except
            log.exception('Failed to list open workflows')
            if ignore_error:
                break
            else:
                raise

        for wf_exec in resp.executions:
            if wf_exec.task_queue.endswith(task_queue_suffix):
                try:
                    await client.workflow_service.terminate_workflow_execution(
                        wsv1.TerminateWorkflowExecutionRequest(
                            namespace=namespace,
                            workflow_execution=wf_exec.execution,
                            reason=f'Terminated by {getpass.getuser()}'),
                        retry=True)
                except Exception:  # pylint: disable=broad-except
                    wf_id = wf_exec.execution.workflow_id
                    log.exception('Failed to terminate workflow: %s', wf_id)
                    if not ignore_error:
                        raise
                log.info('Terminated: %s', wf_exec.execution)
        next_page_token = resp.next_page_token
        if not resp.executions or not next_page_token:
            break


async def main():
    await run(ignore_error=False)


if __name__ == "__main__":
    asyncio.run(main())
