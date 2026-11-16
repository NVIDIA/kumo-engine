# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "subscription_id" {
  type = string
}
variable "deployment_name" {
  type = string
}
variable "image_id" {
  type        = string
  description = "Azure managed image or Shared Image Gallery version resource ID."
}
variable "admin_ssh_public_key" {
  type = string
}
variable "location" {
  type    = string
  default = "westus2"
}
variable "resource_group_name" {
  type    = string
  default = ""
}
variable "storage_account_name" {
  type    = string
  default = ""
  validation {
    condition     = var.storage_account_name == "" || can(regex("^[a-z0-9]{3,24}$", var.storage_account_name))
    error_message = "storage_account_name must be 3-24 lowercase alphanumeric characters."
  }
}
variable "azure_environment" {
  type    = string
  default = "public"
  validation {
    condition     = contains(["public", "usgovernment", "china"], var.azure_environment)
    error_message = "azure_environment must be public, usgovernment, or china."
  }
}
variable "dataplane_filesystem" {
  type    = string
  default = "dataplane"
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.dataplane_filesystem))
    error_message = "dataplane_filesystem must be a valid 3-63 character Azure container name."
  }
}
variable "dataplane_prefix" {
  type    = string
  default = "kumo/dataplane"
}
variable "artifact_filesystem" {
  type    = string
  default = "artifacts"
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.artifact_filesystem))
    error_message = "artifact_filesystem must be a valid 3-63 character Azure container name."
  }
}
variable "artifact_prefix" {
  type    = string
  default = "kumo"
}
variable "temporary_object_retention_days" {
  type    = number
  default = 7
  validation {
    condition     = var.temporary_object_retention_days >= 1
    error_message = "temporary_object_retention_days must be at least 1."
  }
}
variable "storage_replication_type" {
  type    = string
  default = "LRS"
}
variable "enable_public_storage_access" {
  type    = bool
  default = false
}
variable "create_storage_private_endpoints" {
  type    = bool
  default = true
}
variable "enable_public_ip" {
  type    = bool
  default = false
}
variable "allowed_ingress_cidrs" {
  type    = list(string)
  default = []
}
variable "vnet_cidr" {
  type    = string
  default = "10.20.0.0/16"
}
variable "subnet_cidr" {
  type    = string
  default = "10.20.1.0/24"
}
variable "vm_size" {
  type    = string
  default = "Standard_NC16as_T4_v3"
}
variable "admin_username" {
  type    = string
  default = "ubuntu"
}
variable "os_disk_type" {
  type    = string
  default = "Premium_LRS"
}
variable "os_disk_size_gb" {
  type    = number
  default = 128
}
variable "enable_tls" {
  type    = string
  default = ""
}
variable "public_url" {
  type    = string
  default = ""
}
variable "disable_auth" {
  type    = string
  default = ""
}
variable "oidc_issuer" {
  type    = string
  default = ""
}
variable "oidc_client_id" {
  type    = string
  default = ""
}
variable "key_vault_id" {
  type        = string
  default     = ""
  description = "Existing Key Vault resource ID containing Direct Connect bootstrap secrets."
}
variable "oidc_client_secret_uri" {
  type        = string
  default     = ""
  description = "Versioned Azure Key Vault secret ID; the value is fetched with managed identity at first boot."
}
variable "direct_connect_login_method" {
  type    = string
  default = ""
}
variable "admin_emails" {
  type    = string
  default = ""
}
variable "direct_connect_admin_temp_password_uri" {
  type        = string
  default     = ""
  description = "Versioned Azure Key Vault secret ID for password-mode bootstrap."
}
