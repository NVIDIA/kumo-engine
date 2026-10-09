# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.worker import Worker

from io_utils import kumo_dataclass, logging
from service_framework._raw_deploy_config import (
    CustomTemporalConfig,
    TemporalCloudConfig,
)
from service_framework.deploy_env import DeployStage, TemporalServerLocation
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons import TemporalClient
from temporalio_commons.worker_interceptors import KumoActivityInterceptor


# While we could use multiple parameters in the activity, Temporal strongly
# encourages using a single dataclass instead which can have fields added to it
# in a backwards-compatible way.
@kumo_dataclass
class ComposeGreetingInput:
    greeting: str
    name: str


# Basic activity that logs and does string concatenation
@activity.defn
async def compose_greeting(input: ComposeGreetingInput) -> str:
    activity.logger.info("Running activity with parameter %s", input)
    return f"{input.greeting}, {input.name}!"


# Basic workflow that logs and invokes an activity
@workflow.defn
class GreetingWorkflow:  # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, name: str) -> str:
        workflow.logger.info("Running workflow with parameter %s", name)
        return await workflow.execute_activity(
            compose_greeting,
            ComposeGreetingInput("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
            schedule_to_close_timeout=timedelta(minutes=1),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('-t', '--target', dest='target_address', type=TemporalServerLocation,
        required=False, default=TemporalServerLocation.CLOUD,
        choices=list(TemporalServerLocation), help='Temporal server location')
    add('-n', '--namespace', default='default', type=str, required=False,
        help='Namespace in Temporal to use.')
    return parser.parse_args()


async def main(args) -> None:
    """
    Hello World example to quickly demonstrate how to:
    1) write activity function and workflow class
    2) start a worker that hosts the activity and workflow
    3) spawn a workflow execution

    Example commands to run:
        python temporalio_commons/examples/hello_world.py
        python temporalio_commons/examples/hello_world.py -t local
        python temporalio_commons/examples/hello_world.py -n '<namespace>'
    """
    # Create the client helper.
    orchestration = (TemporalCloudConfig()
                     if args.target_address == TemporalServerLocation.CLOUD
                     else CustomTemporalConfig(
                         server_address='127.0.0.1:7233'))
    temporal_config = DeployEnvConfig(client_id='hello_world_app',
                                      orchestration=orchestration,
                                      deploy_mode=DeployStage.DEV)
    temporal_client = TemporalClient(temporal_config)

    await temporal_client.init()
    client = await temporal_client.get()
    # Run a worker for the workflow with context manager to tear down
    # the worker at the end.

    task_queue = temporal_config.task_queue("hello-world-task-queue")
    async with Worker(client, task_queue=task_queue,
                      workflows=[GreetingWorkflow],
                      activities=[compose_greeting],
                      interceptors=[KumoActivityInterceptor()]):
        # While the worker is running, use the client to run the workflow and
        # print out its result. Note, in many production setups, the client
        # would be in a completely separate process from the worker.
        result = await client.execute_workflow(
            GreetingWorkflow.run,
            "World",
            id="hello-activity-workflow-id",
            task_queue=task_queue,
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    logging.initialize_logging(logging.LoggingMode.DEBUG, "/tmp/")
    asyncio.run(main(parse_args()))
