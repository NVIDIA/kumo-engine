# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "vm_id" {
  value = azurerm_linux_virtual_machine.kumo.id
}
output "private_ip" {
  value = azurerm_network_interface.kumo.private_ip_address
}
output "public_ip" {
  value = var.enable_public_ip ? azurerm_public_ip.kumo[0].ip_address : ""
}
output "managed_identity_client_id" {
  value = azurerm_user_assigned_identity.kumo.client_id
}
output "managed_identity_principal_id" {
  value = azurerm_user_assigned_identity.kumo.principal_id
}
output "dataplane_root" {
  value = local.dataplane_root
}
output "artifact_root" {
  value = local.artifact_root
}
output "storage_account_name" {
  value = azurerm_storage_account.kumo.name
}
output "serial_console_command" {
  value = "az serial-console connect --resource-group ${azurerm_resource_group.kumo.name} --name ${azurerm_linux_virtual_machine.kumo.name}"
}
