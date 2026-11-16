# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
#!/usr/bin/env python3
# Terminate an AWS EC2 instance.
# Copy the script to customer temporal worker and run it there.
# Use it when a customer workflow failed with leftover instances.
import argparse

import boto3


def terminate_instance(session, region, instance_id):
    """Terminate an EC2 instance"""
    ec2_client = session.client('ec2', region_name=region)

    try:
        # First check if the instance exists and get its current state
        response = ec2_client.describe_instances(InstanceIds=[instance_id])

        if not response['Reservations'] or not response['Reservations'][0][
                'Instances']:
            print(f"Instance {instance_id} not found in region {region}")
            return False

        instance = response['Reservations'][0]['Instances'][0]
        current_state = instance['State']['Name']

        if current_state == 'terminated':
            print(f"Instance {instance_id} is already terminated")
            return True

        # Terminate the instance
        response = ec2_client.terminate_instances(InstanceIds=[instance_id])
        print(
            f"Termination initiated for instance {instance_id} in region {region}"
        )

        # Wait for the instance to be terminated
        print("Waiting for instance to be terminated...")
        waiter = ec2_client.get_waiter('instance_terminated')
        waiter.wait(InstanceIds=[instance_id])

        print(f"Instance {instance_id} has been successfully terminated")
        return True

    except ec2_client.exceptions.ClientError as e:
        print(f"Error terminating instance {instance_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Terminate AWS EC2 instances.')
    parser.add_argument(
        '--customer-role-arn', required=False,
        default='arn:aws:iam::786169356362:role/odin/kumo-external-access-role',
        help='Customer\'s IAM role ARN')
    parser.add_argument('--region', required=False, default='us-east-1',
                        help='AWS region of the instance (default: us-east-1)')
    parser.add_argument('--instance-id', required=True,
                        help='ID of the EC2 instance to terminate')
    parser.add_argument(
        '--role-session-name', default='CustomerRoleSession',
        help='Session name when assuming role (default: CustomerRoleSession)')

    args = parser.parse_args()

    # Use STS to assume the customer's role
    sts_client = boto3.client('sts')
    response = sts_client.assume_role(RoleArn=args.customer_role_arn,
                                      RoleSessionName=args.role_session_name)

    credentials = response['Credentials']

    customer_session = boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'])

    terminate_instance(customer_session, args.region, args.instance_id)


if __name__ == "__main__":
    main()
