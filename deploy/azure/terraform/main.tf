# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "= 4.80.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  environment     = var.azure_environment
}

check "secret_references_require_key_vault_rbac" {
  assert {
    condition = (
      var.key_vault_id != "" ||
      (var.oidc_client_secret_uri == "" && var.direct_connect_admin_temp_password_uri == "")
    )
    error_message = "key_vault_id is required when a Direct Connect secret URI is configured."
  }
}

check "storage_account_name_is_valid" {
  assert {
    condition     = can(regex("^[a-z0-9]{3,24}$", local.storage_account_name))
    error_message = "The explicit or deployment-name-derived storage account name must be 3-24 lowercase alphanumeric characters. Set storage_account_name explicitly when the derived name is invalid."
  }
}

locals {
  name = "kumo-${var.deployment_name}"
  storage_dns_suffix = {
    public       = "dfs.core.windows.net"
    usgovernment = "dfs.core.usgovcloudapi.net"
    china        = "dfs.core.chinacloudapi.cn"
  }[var.azure_environment]
  blob_dns_suffix      = replace(local.storage_dns_suffix, "dfs.", "blob.")
  storage_account_name = var.storage_account_name != "" ? var.storage_account_name : substr(replace(lower("kumo${var.deployment_name}"), "/[^0-9a-z]/", ""), 0, 24)
  dataplane_root       = "abfss://${var.dataplane_filesystem}@${local.storage_account_name}.${local.storage_dns_suffix}/${trim(var.dataplane_prefix, "/")}"
  artifact_root        = "abfss://${var.artifact_filesystem}@${local.storage_account_name}.${local.storage_dns_suffix}/${trim(var.artifact_prefix, "/")}"
  temporary_prefixes = [
    "${var.dataplane_filesystem}/${trim(var.dataplane_prefix, "/") != "" ? "${trim(var.dataplane_prefix, "/")}/" : ""}tmp_payloads/",
    "${var.dataplane_filesystem}/${trim(var.dataplane_prefix, "/") != "" ? "${trim(var.dataplane_prefix, "/")}/" : ""}temporary-uploaded-datasets/",
    "${var.dataplane_filesystem}/${trim(var.dataplane_prefix, "/") != "" ? "${trim(var.dataplane_prefix, "/")}/" : ""}tmp_archives/",
  ]
  kumo_config = join("\n", compact([
    "dataplane_root: ${local.dataplane_root}",
    "artifact_root: ${local.artifact_root}",
    "lifecycle_ready: true",
    "azure_cloud: ${var.azure_environment}",
    "azure_client_id: ${azurerm_user_assigned_identity.kumo.client_id}",
    var.enable_tls != "" ? "enable_tls: ${var.enable_tls}" : "",
    var.public_url != "" ? "public_url: ${var.public_url}" : "",
    var.disable_auth != "" ? "disable_auth: ${var.disable_auth}" : "",
    var.oidc_issuer != "" ? "oidc_issuer: ${var.oidc_issuer}" : "",
    var.oidc_client_id != "" ? "oidc_client_id: ${var.oidc_client_id}" : "",
    var.oidc_client_secret_uri != "" ? "oidc_client_secret_uri: ${var.oidc_client_secret_uri}" : "",
    var.direct_connect_login_method != "" ? "direct_connect_login_method: ${var.direct_connect_login_method}" : "",
    var.admin_emails != "" ? "admin_emails: ${var.admin_emails}" : "",
    var.direct_connect_admin_temp_password_uri != "" ? "direct_connect_admin_temp_password_uri: ${var.direct_connect_admin_temp_password_uri}" : "",
  ]))
}

resource "azurerm_resource_group" "kumo" {
  name     = var.resource_group_name != "" ? var.resource_group_name : local.name
  location = var.location
}

resource "azurerm_user_assigned_identity" "kumo" {
  name                = local.name
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name
}

resource "azurerm_storage_account" "kumo" {
  name                            = local.storage_account_name
  resource_group_name             = azurerm_resource_group.kumo.name
  location                        = azurerm_resource_group.kumo.location
  account_tier                    = "Standard"
  account_replication_type        = var.storage_replication_type
  account_kind                    = "StorageV2"
  is_hns_enabled                  = true
  min_tls_version                 = "TLS1_2"
  public_network_access_enabled   = var.enable_public_storage_access
  allow_nested_items_to_be_public = false

  network_rules {
    default_action             = var.enable_public_storage_access ? "Allow" : "Deny"
    bypass                     = ["AzureServices"]
    virtual_network_subnet_ids = var.enable_public_storage_access ? [] : [azurerm_subnet.kumo.id]
  }
}

resource "azurerm_storage_data_lake_gen2_filesystem" "dataplane" {
  name               = var.dataplane_filesystem
  storage_account_id = azurerm_storage_account.kumo.id

  # With public storage disabled, create the private data path before the
  # provider performs filesystem operations.
  depends_on = [
    azurerm_private_endpoint.storage,
    azurerm_private_dns_zone_virtual_network_link.storage,
  ]
}

resource "azurerm_storage_data_lake_gen2_filesystem" "artifacts" {
  name               = var.artifact_filesystem
  storage_account_id = azurerm_storage_account.kumo.id

  depends_on = [
    azurerm_private_endpoint.storage,
    azurerm_private_dns_zone_virtual_network_link.storage,
  ]
}

resource "azurerm_storage_management_policy" "temporary_objects" {
  storage_account_id = azurerm_storage_account.kumo.id

  rule {
    name    = "expire-kumo-temporary-objects"
    enabled = true
    filters {
      prefix_match = local.temporary_prefixes
      blob_types   = ["blockBlob"]
    }
    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = var.temporary_object_retention_days
      }
    }
  }
}

# Data Contributor provides scoped list/read/write/delete on the dataplane.
resource "azurerm_role_assignment" "dataplane_contributor" {
  scope                = "${azurerm_storage_account.kumo.id}/blobServices/default/containers/${azurerm_storage_data_lake_gen2_filesystem.dataplane.name}"
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.kumo.principal_id
}

# Runtime artifacts are immutable inputs for the VM identity.
resource "azurerm_role_assignment" "artifact_reader" {
  scope                = "${azurerm_storage_account.kumo.id}/blobServices/default/containers/${azurerm_storage_data_lake_gen2_filesystem.artifacts.name}"
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.kumo.principal_id
}

# Required for user-delegation SAS URLs; scoped to this storage account.
resource "azurerm_role_assignment" "blob_delegator" {
  scope                = azurerm_storage_account.kumo.id
  role_definition_name = "Storage Blob Delegator"
  principal_id         = azurerm_user_assigned_identity.kumo.principal_id
}

# Optional external Key Vault contains Direct Connect bootstrap secrets. Only
# secret identifiers, never values, are placed in IMDS user data or tfvars.
resource "azurerm_role_assignment" "key_vault_secret_reader" {
  count                = var.key_vault_id != "" ? 1 : 0
  scope                = var.key_vault_id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.kumo.principal_id
}

resource "azurerm_virtual_network" "kumo" {
  name                = local.name
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name
}

resource "azurerm_subnet" "kumo" {
  name                 = "kumo"
  resource_group_name  = azurerm_resource_group.kumo.name
  virtual_network_name = azurerm_virtual_network.kumo.name
  address_prefixes     = [var.subnet_cidr]
  service_endpoints    = ["Microsoft.Storage"]
}

resource "azurerm_network_security_group" "kumo" {
  name                = local.name
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name

  dynamic "security_rule" {
    for_each = var.allowed_ingress_cidrs
    content {
      name                       = "https-${security_rule.key}"
      priority                   = 100 + security_rule.key
      direction                  = "Inbound"
      access                     = "Allow"
      protocol                   = "Tcp"
      source_port_range          = "*"
      destination_port_range     = var.enable_tls == "true" ? "443" : "80"
      source_address_prefix      = security_rule.value
      destination_address_prefix = "*"
    }
  }
}

resource "azurerm_subnet_network_security_group_association" "kumo" {
  subnet_id                 = azurerm_subnet.kumo.id
  network_security_group_id = azurerm_network_security_group.kumo.id
}

resource "azurerm_public_ip" "kumo" {
  count               = var.enable_public_ip ? 1 : 0
  name                = local.name
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

resource "azurerm_network_interface" "kumo" {
  name                = local.name
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name
  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.kumo.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = var.enable_public_ip ? azurerm_public_ip.kumo[0].id : null
  }
}

resource "azurerm_private_endpoint" "storage" {
  for_each = var.create_storage_private_endpoints ? toset(["blob", "dfs"]) : []

  name                = "${local.name}-${each.key}"
  location            = azurerm_resource_group.kumo.location
  resource_group_name = azurerm_resource_group.kumo.name
  subnet_id           = azurerm_subnet.kumo.id
  private_service_connection {
    name                           = "${local.name}-${each.key}"
    private_connection_resource_id = azurerm_storage_account.kumo.id
    subresource_names              = [each.key]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage[each.key].id]
  }
}

resource "azurerm_private_dns_zone" "storage" {
  for_each = var.create_storage_private_endpoints ? toset(["blob", "dfs"]) : []

  name                = "privatelink.${each.key == "blob" ? local.blob_dns_suffix : local.storage_dns_suffix}"
  resource_group_name = azurerm_resource_group.kumo.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "storage" {
  for_each = azurerm_private_dns_zone.storage

  name                  = "${local.name}-${each.key}"
  resource_group_name   = azurerm_resource_group.kumo.name
  private_dns_zone_name = each.value.name
  virtual_network_id    = azurerm_virtual_network.kumo.id
}

resource "azurerm_linux_virtual_machine" "kumo" {
  name                  = local.name
  resource_group_name   = azurerm_resource_group.kumo.name
  location              = azurerm_resource_group.kumo.location
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.kumo.id]
  source_image_id       = var.image_id
  # user_data is retrievable through IMDS by firstboot and is intentionally
  # secret-free. Key Vault identifiers are resolved with managed identity.
  user_data = base64encode(local.kumo_config)

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.admin_ssh_public_key
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.kumo.id]
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = var.os_disk_type
    disk_size_gb         = var.os_disk_size_gb
  }

  # Azure Serial Console requires boot diagnostics. The empty block uses
  # managed boot diagnostics storage and keeps the emitted command usable.
  boot_diagnostics {}

  depends_on = [
    azurerm_role_assignment.dataplane_contributor,
    azurerm_role_assignment.artifact_reader,
    azurerm_role_assignment.blob_delegator,
    azurerm_role_assignment.key_vault_secret_reader,
    azurerm_storage_management_policy.temporary_objects,
  ]
}
