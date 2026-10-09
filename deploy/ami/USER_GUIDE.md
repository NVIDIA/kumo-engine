# Kumo AMI Installation Guide for AWS

This guide describes how to run the Kumo single-host AMI in your own AWS
account. Obtain a compatible AMI available in your account and target region,
then create the AWS resources around it: storage, IAM, networking, load balancing,
DNS, and user access. AMI distribution and public release availability remain
release prerequisites.

The resulting deployment is a single EC2 host running the Kumo UI, REST APIs,
Direct Connect authentication, Temporal workers, Prometheus, Jaeger, and
supporting services through Docker Compose. A typical customer deployment exposes the host
through an internet-facing or internal Application Load Balancer (ALB) and keeps
the EC2 instance in a private subnet.

## Architecture

```text
Users
  |
  | https://kumo.example.com
  v
Route 53 or customer DNS
  |
  v
AWS Application Load Balancer, TLS terminated with ACM
  |
  | HTTP :80, from ALB security group only
  v
EC2 instance launched from the Kumo AMI
  |
  | IAM instance profile
  v
S3 dataplane bucket and source-data buckets
```

All persistent metadata state runs on the instance: a bundled Postgres
container holds the **StateDB** (REST metadata + ML job metadata), stored in a
Docker volume on the root EBS. The diskgraph index lives on the local NVMe
(`/mnt/localssd`) and is rebuilt as needed.

The AMI expects a `public_url` in EC2 user data. For an ALB deployment, set it
to the browser-facing HTTPS origin, for example `https://kumo.example.com`.
The ALB can terminate TLS and forward plain HTTP to the instance on port 80.

## Requirements

Before launching the AMI, gather or create the following:

- A compatible AMI ID that your AWS account can launch in the selected region.
- AWS CLI v2: configured with permissions to manage EC2, ELBv2, IAM, S3, ACM,
  Route 53, and SSM in your account.
- VPC and subnets: two subnets for the ALB, normally public subnets in different
  Availability Zones, plus one subnet for the EC2 instance.
- Host instance type: use an NVIDIA GPU instance with local NVMe. `g4dn.4xlarge`
  is suitable for small POCs; `g4dn.8xlarge` or larger is recommended for
  larger data and 128 GB or more memory.
- Root EBS volume: at least 300 GB gp3. Use 4000 IOPS and 1000 MB/s throughput
  for faster cold boot from the AMI snapshot.
- Dataplane S3 bucket: Kumo internal state, training tables, materialized data,
  model artifacts, and uploaded files should use S3. The bucket also needs a
  CORS policy naming the Kumo hostname, because browsers upload local files to
  it directly.
- Source-data S3 access: the instance profile must be able to read the buckets
  containing your input CSV or Parquet data.
- ALB and ACM certificate: required for normal browser access over HTTPS.
- DNS name: a Route 53 alias record or a CNAME in your own DNS provider.
- Security groups: allow user traffic to the ALB and allow only the ALB to
  reach the instance.
- SSM access: recommended for operational access without SSH. Private subnets
  without NAT need VPC endpoints for SSM.

## Network and Firewall Rules

Use these rules as the baseline:

| Component | Direction | Rule |
| --- | --- | --- |
| ALB security group | Inbound | TCP 443 from approved user CIDRs |
| ALB security group | Inbound | TCP 80 from approved user CIDRs, optional redirect to HTTPS |
| ALB security group | Outbound | TCP 80 to the instance security group |
| Instance security group | Inbound | TCP 80 from the ALB security group only |
| Instance security group | Inbound | No direct SSH required when SSM is enabled |
| Instance security group | Outbound | HTTPS to AWS APIs, S3, SSM endpoints, or NAT |
| End-user network | Outbound | HTTPS 443 to the regional S3 endpoint, for local file upload |

If the instance is in a private subnet without NAT, create these VPC endpoints:

- S3 gateway endpoint on the private route table.
- Interface endpoints for `ssm`, `ssmmessages`, and `ec2messages`.

Docker images are baked into the AMI, so the instance should not need public
internet access for image pulls during normal boot.

## Step-by-Step AWS CLI Setup

The commands below assume an existing VPC, subnets, ACM certificate, and DNS
zone. Replace all placeholder values before running them.

### 1. Set Environment Variables

```bash
export AWS_PROFILE=<your-aws-profile>
export AWS_REGION=us-west-2

export DEPLOYMENT_NAME=acme-bank
export DNS_NAME=kumo.acme.example
export USER_CIDR=203.0.113.0/24

export KUMO_AMI_ID=ami-0123456789abcdef0
export INSTANCE_TYPE=g4dn.4xlarge

export VPC_ID=vpc-0123456789abcdef0
export INSTANCE_SUBNET_ID=subnet-private123456789
export ALB_SUBNET_IDS="subnet-public111111111 subnet-public222222222"
export PRIVATE_ROUTE_TABLE_ID=rtb-0123456789abcdef0

export CERT_ARN=arn:aws:acm:us-west-2:111122223333:certificate/00000000-0000-0000-0000-000000000000
export HOSTED_ZONE_ID=Z0123456789ABCDEF

export DATAPLANE_BUCKET=acme-kumo-dataplane
export SOURCE_BUCKET=acme-source-data

export INSTANCE_ROLE=KumoSingleHostRole
export INSTANCE_PROFILE=KumoSingleHostInstanceProfile
```

### 2. Confirm AMI Availability

```bash
aws ec2 describe-images \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --image-ids "$KUMO_AMI_ID" \
  --query 'Images[0].{ImageId:ImageId,Name:Name,Owner:OwnerId,State:State,Created:CreationDate}' \
  --output table
```

If this returns `InvalidAMIID.NotFound`, verify the image ID, region, and
launch permissions with the image owner.

### 3. Create the Dataplane S3 Bucket

For regions other than `us-east-1`:

```bash
aws s3api create-bucket \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --bucket "$DATAPLANE_BUCKET" \
  --create-bucket-configuration LocationConstraint="$AWS_REGION"
```

For `us-east-1`, omit `--create-bucket-configuration`.

Enable encryption and block public access:

```bash
aws s3api put-public-access-block \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --bucket "$DATAPLANE_BUCKET" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-encryption \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --bucket "$DATAPLANE_BUCKET" \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# Kumo enables Temporal payload offload only when these scoped expiry rules
# are explicitly declared ready in first-boot config.
aws s3api put-bucket-lifecycle-configuration \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --bucket "$DATAPLANE_BUCKET" \
  --lifecycle-configuration '{"Rules":[{"ID":"ExpireKumoPayloads","Status":"Enabled","Filter":{"Prefix":"kumo/tmp_payloads/"},"Expiration":{"Days":7}},{"ID":"ExpireKumoUploads","Status":"Enabled","Filter":{"Prefix":"kumo/temporary-uploaded-datasets/"},"Expiration":{"Days":7}},{"ID":"ExpireKumoArchives","Status":"Enabled","Filter":{"Prefix":"kumo/tmp_archives/"},"Expiration":{"Days":7}}]}'
```

Local file upload is the one flow where the browser `PUT`s to S3 itself, so the
bucket also needs a CORS policy naming the Kumo hostname. Without it the browser
blocks the upload before any bytes are sent and the UI reports only
`Could not upload the file!`, while every server-side operation keeps working.
`ExposeHeaders` must list `ETag`: the UI reads it from each part to complete the
multipart upload, and browsers hide unexposed response headers.

```bash
aws s3api put-bucket-cors \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --bucket "$DATAPLANE_BUCKET" \
  --cors-configuration '{"CORSRules":[{"AllowedOrigins":["https://'"$DNS_NAME"'"],"AllowedMethods":["PUT","POST","GET","HEAD"],"AllowedHeaders":["*"],"ExposeHeaders":["ETag"],"MaxAgeSeconds":3600}]}'
```

`put-bucket-cors` replaces the entire CORS configuration rather than adding to
it. If the bucket is shared with anything else, read the current rules with
`aws s3api get-bucket-cors` first and submit the merged set.

For an existing AWS deployment, install and verify these lifecycle rules
before replacing the instance or adding `lifecycle_ready: true` to user data.
Upgrades that omit the field default it to `false`; Kumo then keeps Temporal
payloads inline/compressed instead of offloading them, even if an older
deployment previously used payload offload. Leave the field false when the
scoped expiry rules are not present.

Your source data bucket can be an existing bucket. The instance profile only
needs read access to source data, but it needs read/write/delete access to the
dataplane bucket.

### 4. Create the Instance IAM Role

Create the EC2 trust policy:

```bash
cat > trust-policy.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON
```

Create the role and attach SSM:

```bash
aws iam create-role \
  --profile "$AWS_PROFILE" \
  --role-name "$INSTANCE_ROLE" \
  --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
  --profile "$AWS_PROFILE" \
  --role-name "$INSTANCE_ROLE" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```

Create an S3 policy for dataplane read/write and source-data read:

```bash
cat > kumo-s3-policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSourceData",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::${SOURCE_BUCKET}"
    },
    {
      "Sid": "ReadSourceObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::${SOURCE_BUCKET}/*"
    },
    {
      "Sid": "UseDataplaneBucket",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads"
      ],
      "Resource": "arn:aws:s3:::${DATAPLANE_BUCKET}"
    },
    {
      "Sid": "UseDataplaneObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "arn:aws:s3:::${DATAPLANE_BUCKET}/*"
    }
  ]
}
JSON

aws iam put-role-policy \
  --profile "$AWS_PROFILE" \
  --role-name "$INSTANCE_ROLE" \
  --policy-name KumoSingleHostS3Access \
  --policy-document file://kumo-s3-policy.json
```

Create the instance profile:

```bash
aws iam create-instance-profile \
  --profile "$AWS_PROFILE" \
  --instance-profile-name "$INSTANCE_PROFILE"

aws iam add-role-to-instance-profile \
  --profile "$AWS_PROFILE" \
  --instance-profile-name "$INSTANCE_PROFILE" \
  --role-name "$INSTANCE_ROLE"
```

Wait a minute for IAM propagation before launching the instance.

If your buckets use a customer-managed KMS key, also grant this role
`kms:Decrypt`, `kms:Encrypt`, `kms:GenerateDataKey`, and `kms:DescribeKey` on
that key.

### 5. Create Security Groups

```bash
export ALB_SG_ID=$(
  aws ec2 create-security-group \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --group-name "${DEPLOYMENT_NAME}-kumo-alb-sg" \
    --description "Kumo ALB security group" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' \
    --output text
)

export HOST_SG_ID=$(
  aws ec2 create-security-group \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --group-name "${DEPLOYMENT_NAME}-kumo-host-sg" \
    --description "Kumo single-host EC2 security group" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' \
    --output text
)
```

Allow users to reach the ALB:

```bash
aws ec2 authorize-security-group-ingress \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-id "$ALB_SG_ID" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=${USER_CIDR},Description='Kumo HTTPS users'}]" \
    "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=${USER_CIDR},Description='Kumo HTTP redirect'}]"
```

Allow the ALB to reach nginx on the instance:

```bash
aws ec2 authorize-security-group-ingress \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-id "$HOST_SG_ID" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=80,ToPort=80,UserIdGroupPairs=[{GroupId=${ALB_SG_ID},Description='Kumo ALB to host'}]"
```

### 6. Add VPC Endpoints for Private Subnets Without NAT

If the instance subnet has NAT egress, this step is optional. If it does not,
create a gateway endpoint for S3:

```bash
aws ec2 create-vpc-endpoint \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --vpc-id "$VPC_ID" \
  --service-name "com.amazonaws.${AWS_REGION}.s3" \
  --vpc-endpoint-type Gateway \
  --route-table-ids "$PRIVATE_ROUTE_TABLE_ID"
```

Create SSM interface endpoints:

```bash
export VPCE_SG_ID=$(
  aws ec2 create-security-group \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --group-name "${DEPLOYMENT_NAME}-kumo-vpce-sg" \
    --description "Kumo VPC endpoint security group" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' \
    --output text
)

aws ec2 authorize-security-group-ingress \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --group-id "$VPCE_SG_ID" \
  --protocol tcp \
  --port 443 \
  --source-group "$HOST_SG_ID"

for service in ssm ssmmessages ec2messages; do
  aws ec2 create-vpc-endpoint \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --vpc-id "$VPC_ID" \
    --service-name "com.amazonaws.${AWS_REGION}.${service}" \
    --vpc-endpoint-type Interface \
    --subnet-ids "$INSTANCE_SUBNET_ID" \
    --security-group-ids "$VPCE_SG_ID" \
    --private-dns-enabled
done
```

### 7. Create EC2 User Data

The user-data format is simple `key: value` pairs. Values should not contain
spaces.

```bash
cat > user-data.yaml <<EOF
dataplane_root: s3://${DATAPLANE_BUCKET}/kumo
lifecycle_ready: true
public_url: https://${DNS_NAME}
enable_tls: false
direct_connect_login_method: password
admin_emails: admin@example.com
direct_connect_admin_temp_password: ChangeThisTemporaryPassword1!
EOF
```

`enable_tls: false` is correct when the ALB terminates TLS and forwards HTTP to
the instance. Set `enable_tls: true` only if the instance itself should serve
HTTPS with a self-signed or customer-provided certificate.

For Direct Connect OIDC mode, replace the password-mode fields with your IdP
settings:

```yaml
direct_connect_login_method: oidc
admin_emails: admin@example.com
oidc_issuer: https://idp.example.com
oidc_client_id: kumo
oidc_client_secret: <client-secret>
oidc_groups_claim: groups
```

### 8. Launch the EC2 Instance

> **Required**: `--metadata-options HttpPutResponseHopLimit=2`. The Kumo
> containers run on a docker bridge network and reach the EC2 Instance
> Metadata Service (`169.254.169.254`) through one extra hop. The IMDSv2
> default hop limit of 1 drops the response, so the in-container AWS SDK
> fails to load instance-role credentials and graph ingestion errors with
> `Failed to load AWS credentials: the credential provider was not enabled`.
> Setting hop limit to 2 is the standard AWS-recommended value for
> containerized workloads.

```bash
export INSTANCE_ID=$(
  aws ec2 run-instances \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --image-id "$KUMO_AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
    --subnet-id "$INSTANCE_SUBNET_ID" \
    --security-group-ids "$HOST_SG_ID" \
    --no-associate-public-ip-address \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2 \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Iops":4000,"Throughput":1000,"DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${DEPLOYMENT_NAME}-kumo-singlehost},{Key=Application,Value=Kumo}]" \
    --user-data file://user-data.yaml \
    --query 'Instances[0].InstanceId' \
    --output text
)

aws ec2 wait instance-running \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID"
```

First boot usually takes several minutes. The first-boot script validates S3
dataplane access before starting services.

### 9. Create the ALB Target Group

```bash
export TG_ARN=$(
  aws elbv2 create-target-group \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --name "${DEPLOYMENT_NAME}-kumo-tg" \
    --protocol HTTP \
    --port 80 \
    --vpc-id "$VPC_ID" \
    --target-type instance \
    --health-check-protocol HTTP \
    --health-check-path /api/health \
    --matcher HttpCode=200-399 \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text
)

aws elbv2 register-targets \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --target-group-arn "$TG_ARN" \
  --targets "Id=${INSTANCE_ID},Port=80"
```

### 10. Create the ALB and Listeners

```bash
export ALB_ARN=$(
  aws elbv2 create-load-balancer \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --name "${DEPLOYMENT_NAME}-kumo-alb" \
    --type application \
    --scheme internet-facing \
    --subnets $ALB_SUBNET_IDS \
    --security-groups "$ALB_SG_ID" \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text
)

aws elbv2 create-listener \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --load-balancer-arn "$ALB_ARN" \
  --protocol HTTPS \
  --port 443 \
  --certificates "CertificateArn=${CERT_ARN}" \
  --default-actions "Type=forward,TargetGroupArn=${TG_ARN}"

aws elbv2 create-listener \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --load-balancer-arn "$ALB_ARN" \
  --protocol HTTP \
  --port 80 \
  --default-actions "Type=redirect,RedirectConfig={Protocol=HTTPS,Port=443,StatusCode=HTTP_301}"
```

Wait for the load balancer to become active:

```bash
aws elbv2 wait load-balancer-available \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --load-balancer-arns "$ALB_ARN"
```

### 11. Configure DNS

For Route 53, create an alias record to the ALB:

```bash
export ALB_DNS_NAME=$(
  aws elbv2 describe-load-balancers \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --load-balancer-arns "$ALB_ARN" \
    --query 'LoadBalancers[0].DNSName' \
    --output text
)

export ALB_ZONE_ID=$(
  aws elbv2 describe-load-balancers \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --load-balancer-arns "$ALB_ARN" \
    --query 'LoadBalancers[0].CanonicalHostedZoneId' \
    --output text
)

cat > route53-change.json <<EOF
{
  "Comment": "Point Kumo DNS to the ALB",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "${DNS_NAME}",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "${ALB_ZONE_ID}",
          "DNSName": "dualstack.${ALB_DNS_NAME}",
          "EvaluateTargetHealth": true
        }
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
  --profile "$AWS_PROFILE" \
  --hosted-zone-id "$HOSTED_ZONE_ID" \
  --change-batch file://route53-change.json
```

If you do not use Route 53, create a CNAME from `DNS_NAME` to `ALB_DNS_NAME`
with your DNS provider.

### 12. Validate the Deployment

Check target health:

```bash
aws elbv2 describe-target-health \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --target-group-arn "$TG_ARN" \
  --query 'TargetHealthDescriptions[*].{Target:Target.Id,Port:Target.Port,State:TargetHealth.State,Reason:TargetHealth.Reason}' \
  --output table
```

Check the public endpoints:

```bash
curl -fsS "https://${DNS_NAME}/api/health"
curl -I "https://${DNS_NAME}/"
```

Check host status through SSM:

```bash
aws ssm send-command \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment kumo-status \
  --parameters commands='["sudo docker ps --format \"{{.Names}} {{.Status}} {{.Image}}\" | sort","curl -fsS http://localhost/api/health"]'
```

You can also start an interactive SSM session:

```bash
aws ssm start-session \
  --profile "$AWS_PROFILE" \
  --region "$AWS_REGION" \
  --target "$INSTANCE_ID"
```

Useful host-side commands:

```bash
kumo-health
kumo-diag --since 2h
sudo docker ps
sudo docker logs kumo-rest --tail 200
sudo tail -100 /var/log/cloud-init-output.log
```

## Default Accounts and First Login

The AMI uses Direct Connect authentication. First boot reads the Direct Connect
auth settings from EC2 user data and stores sensitive values in the local Kumo
secret file.

### Password Mode

When `direct_connect_login_method: password` is set, first boot creates local
Kumo admin users for each email in `admin_emails`. Each seeded admin receives
the `direct_connect_admin_temp_password` value and must set a new password on
first login.

1. Open `https://<your-dns-name>/`.
2. Log in with one of the `admin_emails` values.
3. Use the temporary password from user data.
4. Set a permanent password when prompted.
5. Continue into the Kumo UI.

Do not leave temporary passwords in use. Set a unique temporary password in EC2
user data before first boot and rotate it after login.

### OIDC Mode

When `direct_connect_login_method: oidc` is set, Kumo redirects users to the
configured OIDC issuer. Register the exact callback
`<browser-origin>/v0/auth/callback`, using the origin of the Kumo `public_url`.
The IdP must issue a refresh token (Kumo requests `offline_access`). Allow
server egress to its discovery, token, and JWKS endpoints; private S3 and SSM
endpoints alone do not provide that access.

An eligible login bootstraps an admin only when no active admin exists, the
normalized email appears in `admin_emails`, and the IdP reports
`email_verified: true`. Manage additional admins in the Kumo admin UI.

1. Open `https://<your-dns-name>/`.
2. Complete the IdP login flow.
3. Return to the Kumo UI.

### Create New Users

Use **Admin > Users** in the Kumo UI to create or update local users. In password
mode, admins can create users and set temporary passwords from the UI. In OIDC
mode, users are linked to IdP identities when they log in, and Kumo role/admin
state is managed in the Kumo admin UI.

Changing a user's role or disabling a user ends that user's active sessions in
both modes, so the user must log in again under the updated permissions.

## Operations Notes

- Stop/start: EC2 stop preserves EBS and S3 state, but local NVMe data is
  ephemeral and is rebuilt when needed. Boot cleanup attempts to terminate
  running workflows returned by the Temporal query in the configured namespace;
  do not assume in-flight jobs resume after a restart.
- State DB: REST/ML metadata runs in a bundled Postgres whose data lives in a
  Docker volume on the root EBS, so it is preserved across stop/start. There is
  no managed point-in-time recovery; back it up with periodic root-EBS
  snapshots. Losing the root volume loses all metadata. External/managed
  Postgres can be configured at first boot with the BYO Postgres user-data
  fields.
- Dataplane: S3 is recommended. Local EBS dataplane is possible but requires
  manual disk management.
- Health: the public REST health endpoint is `https://<dns>/api/health`.
- Diagnostics: `kumo-diag` creates a support bundle under `/tmp`.
- Logs: Docker logs are on the host and first-boot logs are in
  `/var/log/cloud-init-output.log`.
- Concurrency: single-host deployments are intended for POC and small
  single-tenant workloads.

## Example CloudFormation

This sample demonstrates the same setup against an existing VPC, subnets, ACM
certificate, and Route 53 hosted zone. It creates the dataplane bucket, IAM
role/profile, security groups, EC2 instance, ALB, target group, listeners, and
DNS record.

Adjust it for your network standards before production use. If you already have
a dataplane bucket, remove the `KumoDataplaneBucket` resource and update the IAM
policy to reference the existing bucket.

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Example Kumo single-host AMI deployment in an existing VPC.

Parameters:
  DeploymentName:
    Type: String
    Default: acme-bank
  KumoAmiId:
    Type: AWS::EC2::Image::Id
  InstanceType:
    Type: String
    Default: g4dn.4xlarge
  VpcId:
    Type: AWS::EC2::VPC::Id
  InstanceSubnetId:
    Type: AWS::EC2::Subnet::Id
  AlbSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
  UserCidr:
    Type: String
    Default: 203.0.113.0/24
  CertificateArn:
    Type: String
  HostedZoneId:
    Type: AWS::Route53::HostedZone::Id
  DNSName:
    Type: String
    Default: kumo.example.com
  DataplaneBucketName:
    Type: String
  SourceDataBucketName:
    Type: String
  AdminEmail:
    Type: String
    Default: admin@example.com
  DefaultUserTempPassword:
    Type: String
    NoEcho: true
    MinLength: 12

Resources:
  KumoDataplaneBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Ref DataplaneBucketName
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      CorsConfiguration:
        CorsRules:
          - AllowedOrigins:
              - !Sub 'https://${DNSName}'
            AllowedMethods:
              - PUT
              - POST
              - GET
              - HEAD
            AllowedHeaders:
              - '*'
            ExposedHeaders:
              - ETag
            MaxAge: 3600
      LifecycleConfiguration:
        Rules:
          - Id: ExpireKumoPayloads
            Status: Enabled
            Prefix: kumo/tmp_payloads/
            ExpirationInDays: 7
          - Id: ExpireKumoUploads
            Status: Enabled
            Prefix: kumo/temporary-uploaded-datasets/
            ExpirationInDays: 7
          - Id: ExpireKumoArchives
            Status: Enabled
            Prefix: kumo/tmp_archives/
            ExpirationInDays: 7

  KumoInstanceRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - !Sub arn:${AWS::Partition}:iam::aws:policy/AmazonSSMManagedInstanceCore
      Policies:
        - PolicyName: KumoSingleHostS3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetBucketLocation
                  - s3:ListBucket
                Resource: !Sub arn:${AWS::Partition}:s3:::${SourceDataBucketName}
              - Effect: Allow
                Action:
                  - s3:GetObject
                Resource: !Sub arn:${AWS::Partition}:s3:::${SourceDataBucketName}/*
              - Effect: Allow
                Action:
                  - s3:GetBucketLocation
                  - s3:ListBucket
                  - s3:ListBucketMultipartUploads
                Resource: !Sub arn:${AWS::Partition}:s3:::${DataplaneBucketName}
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:AbortMultipartUpload
                  - s3:ListMultipartUploadParts
                Resource: !Sub arn:${AWS::Partition}:s3:::${DataplaneBucketName}/*

  KumoInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref KumoInstanceRole

  KumoAlbSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Kumo ALB security group
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: !Ref UserCidr
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: !Ref UserCidr

  KumoHostSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Kumo EC2 host security group
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupId: !Ref KumoAlbSecurityGroup

  KumoInstance:
    Type: AWS::EC2::Instance
    DependsOn:
      - KumoDataplaneBucket
      - KumoInstanceProfile
    Properties:
      ImageId: !Ref KumoAmiId
      InstanceType: !Ref InstanceType
      IamInstanceProfile: !Ref KumoInstanceProfile
      MetadataOptions:
        HttpEndpoint: enabled
        HttpTokens: required
        HttpPutResponseHopLimit: 2
      NetworkInterfaces:
        - DeviceIndex: '0'
          SubnetId: !Ref InstanceSubnetId
          GroupSet:
            - !Ref KumoHostSecurityGroup
          AssociatePublicIpAddress: false
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
            VolumeSize: 300
            VolumeType: gp3
            Iops: 4000
            Throughput: 1000
            DeleteOnTermination: true
      UserData:
        Fn::Base64: !Sub |
          dataplane_root: s3://${DataplaneBucketName}/kumo
          lifecycle_ready: true
          public_url: https://${DNSName}
          enable_tls: false
          direct_connect_login_method: password
          admin_emails: ${AdminEmail}
          direct_connect_admin_temp_password: ${DefaultUserTempPassword}
      Tags:
        - Key: Name
          Value: !Sub ${DeploymentName}-kumo-singlehost
        - Key: Application
          Value: Kumo

  KumoTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub ${DeploymentName}-kumo-tg
      VpcId: !Ref VpcId
      Protocol: HTTP
      Port: 80
      TargetType: instance
      HealthCheckProtocol: HTTP
      HealthCheckPath: /api/health
      Matcher:
        HttpCode: 200-399
      Targets:
        - Id: !Ref KumoInstance
          Port: 80

  KumoLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub ${DeploymentName}-kumo-alb
      Scheme: internet-facing
      Type: application
      Subnets: !Ref AlbSubnetIds
      SecurityGroups:
        - !Ref KumoAlbSecurityGroup

  KumoHttpsListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref KumoLoadBalancer
      Protocol: HTTPS
      Port: 443
      Certificates:
        - CertificateArn: !Ref CertificateArn
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref KumoTargetGroup

  KumoHttpRedirectListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref KumoLoadBalancer
      Protocol: HTTP
      Port: 80
      DefaultActions:
        - Type: redirect
          RedirectConfig:
            Protocol: HTTPS
            Port: '443'
            StatusCode: HTTP_301

  KumoDnsRecord:
    Type: AWS::Route53::RecordSet
    Properties:
      HostedZoneId: !Ref HostedZoneId
      Name: !Ref DNSName
      Type: A
      AliasTarget:
        HostedZoneId: !GetAtt KumoLoadBalancer.CanonicalHostedZoneID
        DNSName: !GetAtt KumoLoadBalancer.DNSName
        EvaluateTargetHealth: true

Outputs:
  KumoUrl:
    Value: !Sub https://${DNSName}/
  InstanceId:
    Value: !Ref KumoInstance
  LoadBalancerDnsName:
    Value: !GetAtt KumoLoadBalancer.DNSName
  TargetGroupArn:
    Value: !Ref KumoTargetGroup
  DataplaneBucket:
    Value: !Ref KumoDataplaneBucket
```
