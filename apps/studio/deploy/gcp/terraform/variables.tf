# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "project" {
  type        = string
  description = "GCP project ID."
}

variable "deployment_name" {
  type        = string
  description = "Deployment identifier. Used as the instance name and to derive the default dataplane bucket."
}

variable "image_name" {
  type        = string
  description = "Kumo singlehost GCP image name, e.g. kumo-singlehost-861aa1b8."
}

variable "zone" {
  type        = string
  default     = "us-west1-b"
  description = "GCP zone for the instance."
}

variable "machine_type" {
  type    = string
  default = "n1-standard-32"
}

variable "accelerator_type" {
  type        = string
  default     = "nvidia-tesla-t4"
  description = "GPU type: nvidia-tesla-t4 (n1), nvidia-l4 (g2), nvidia-tesla-a100 (a2)."
}

variable "accelerator_count" {
  type    = number
  default = 1
}

variable "boot_disk_size_gb" {
  type    = number
  default = 100
}

variable "boot_disk_type" {
  type    = string
  default = "pd-ssd"
}

variable "local_ssd_count" {
  type        = number
  default     = 1
  description = "Number of 375 GB NVMe local SSDs to attach."
}

variable "local_ssd_interface" {
  type    = string
  default = "NVME"
  validation {
    condition     = contains(["NVME", "SCSI"], var.local_ssd_interface)
    error_message = "local_ssd_interface must be NVME or SCSI."
  }
}

variable "dataplane_root" {
  type        = string
  default     = ""
  description = "GCS dataplane root, e.g. gs://my-bucket/kumo/dataplane. Defaults to gs://kumo-<deployment_name>/kumo/dataplane."
}

variable "temporary_object_retention_days" {
  type        = number
  default     = 7
  description = "Days before objects under tmp_payloads/, temporary-uploaded-datasets/, and tmp_archives/ expire."

  validation {
    condition     = var.temporary_object_retention_days >= 1
    error_message = "temporary_object_retention_days must be at least 1."
  }
}

variable "enable_tls" {
  type        = string
  default     = ""
  description = "Enable TLS with self-signed cert. Leave empty to use the image default (false)."
}

variable "public_url" {
  type        = string
  default     = ""
  description = "Browser-facing Kumo origin for auth-backed singlehost deployments."
}

variable "allow_local_public_url" {
  type        = string
  default     = ""
  description = "Allow localhost public_url for local smoke tests."
}

variable "disable_auth" {
  type        = string
  default     = ""
  description = "Disable auth for testing only."
}

variable "oidc_issuer" {
  type        = string
  default     = ""
  description = "Direct Connect OIDC issuer URL."
}

variable "oidc_client_id" {
  type        = string
  default     = ""
  description = "Direct Connect OIDC client ID."
}

variable "oidc_client_secret" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Direct Connect OIDC client secret."
}

variable "oidc_groups_claim" {
  type        = string
  default     = ""
  description = "Direct Connect OIDC groups claim. Leave empty for groups."
}

variable "direct_connect_login_method" {
  type        = string
  default     = ""
  description = "Direct Connect login mode: oidc or password."
}

variable "admin_emails" {
  type        = string
  default     = ""
  description = "Comma-separated Direct Connect bootstrap admin emails."
}

variable "direct_connect_admin_temp_password" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Temporary first-login password for Direct Connect password mode."
}

variable "subnet" {
  type        = string
  default     = ""
  description = "Full subnet self-link, e.g. projects/PROJECT/regions/REGION/subnetworks/NAME. Leave empty to use the default network."
}

variable "service_account_email" {
  type        = string
  default     = ""
  description = "Existing service account email to attach. Leave empty to auto-create kumo-<deployment_name>@<project>.iam.gserviceaccount.com."
}

variable "instance_name" {
  type        = string
  default     = ""
  description = "Instance name. Defaults to deployment_name."
}

variable "network" {
  type        = string
  default     = "default"
  description = "VPC network name or self-link. Used when subnet is not specified."
}

variable "enable_public_ip" {
  type        = bool
  default     = false
  description = "Assign an ephemeral public IP to the instance. When false (default), the instance has no external IP and requires Private Google Access for GCS and IAP for SSH."
}

variable "create_subnet" {
  type        = bool
  default     = false
  description = "Create a new subnet with Private Google Access enabled. Requires subnet_cidr to be set. Use this for fresh deployments where you don't have an existing subnet."
}

variable "subnet_cidr" {
  type        = string
  default     = "10.0.0.0/24"
  description = "CIDR range for the new subnet when create_subnet = true."
}

variable "enable_private_google_access" {
  type        = bool
  default     = false
  description = "Enable Private Google Access on a newly created subnet (create_subnet = true). For existing subnets, enable PGA outside Terraform before applying."
}

variable "create_iap_firewall_rule" {
  type        = bool
  default     = true
  description = "Create a firewall rule allowing IAP TCP forwarding (port 22 from 35.235.240.0/20). Required for gcloud compute ssh --tunnel-through-iap."
}

variable "data_disk_size_gb" {
  type        = number
  default     = 0
  description = "Size in GB of an additional persistent data disk mounted at /data. Set to 0 to skip. Useful when dataplane_root is a local path."
}

variable "data_disk_type" {
  type    = string
  default = "pd-ssd"
}
