# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
#!/usr/bin/env python3
# List AWS EC2 instances.
# Copy the script to customer temporal worker and run it there.
import argparse
import json
from datetime import datetime

import boto3


def list_instances(session, region):
    """List EC2 instances with their details"""
    ec2_client = session.client('ec2', region_name=region)

    try:
        # Get all instances
        response = ec2_client.describe_instances()

        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                # Get instance name from tags, if it exists
                name = "-"
                if 'Tags' in instance:
                    for tag in instance['Tags']:
                        if tag['Key'] == 'Name':
                            name = tag['Value']
                            break

                # Get IP addresses, handling cases where they might not exist
                private_ip = instance.get('PrivateIpAddress', '-')
                public_ip = instance.get('PublicIpAddress', '-')

                # Format launch time
                launch_time = instance['LaunchTime']
                if isinstance(launch_time, datetime):
                    launch_time = launch_time.strftime('%Y-%m-%d %H:%M:%S')

                # Collect instance data
                instance_data = {
                    'InstanceID': instance['InstanceId'],
                    'Name': name,
                    'Type': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'PrivateIP': private_ip,
                    'PublicIP': public_ip,
                    'LaunchTime': launch_time
                }
                instances.append(instance_data)

        return instances

    except ec2_client.exceptions.ClientError as e:
        print(f"Error listing instances: {e}")
        return []


def display_instances(instances):
    """Display instances in a tabular format"""
    if not instances:
        print("No instances found")
        return

    # Prepare table data
    table_data = []
    for instance in instances:
        table_data.append([
            instance['InstanceID'], instance['Name'], instance['Type'],
            instance['State'], instance['PrivateIP'], instance['PublicIP'],
            instance['LaunchTime']
        ])
        print(table_data[-1])

    state_counts = {}
    for instance in instances:
        state = instance['State']
        state_counts[state] = state_counts.get(state, 0) + 1

    print("\nInstance count by state:")
    for state, count in state_counts.items():
        print(f"  {state}: {count}")

    print(f"\nTotal instances: {len(instances)}")


def main():
    parser = argparse.ArgumentParser(description='List AWS EC2 instances.')
    parser.add_argument(
        '--customer-role-arn', required=False,
        default='arn:aws:iam::786169356362:role/odin/kumo-external-access-role',
        help='Customer\'s IAM role ARN')
    parser.add_argument(
        '--region', required=False, default='us-east-1',
        help='AWS region to list instances from (default: us-east-1)')
    parser.add_argument('--output', choices=['table', 'json'], default='table',
                        help='Output format (table or json, default: table)')
    parser.add_argument(
        '--role-session-name', default='CustomerRoleSession',
        help='Session name when assuming role (default: CustomerRoleSession)')
    parser.add_argument(
        '--filter', required=False,
        help='Filter instances by state (e.g., running, stopped)')

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

    # List instances
    instances = list_instances(customer_session, args.region)

    # Apply filter if specified
    if args.filter:
        instances = [
            i for i in instances if i['State'].lower() == args.filter.lower()
        ]

    # Display output
    if args.output == 'json':
        print(json.dumps(instances, indent=2, default=str))
    else:
        display_instances(instances)


if __name__ == "__main__":
    main()
