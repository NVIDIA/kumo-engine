# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

output "instance_name" {
  value = google_compute_instance.kumo.name
}

output "instance_self_link" {
  value = google_compute_instance.kumo.self_link
}

output "public_ip" {
  value = var.enable_public_ip ? google_compute_instance.kumo.network_interface[0].access_config[0].nat_ip : ""
}

output "service_account_email" {
  value = local.sa_email
}

output "dataplane_bucket" {
  value = google_storage_bucket.dataplane.name
}

output "ssh_command" {
  value = "gcloud compute ssh ${google_compute_instance.kumo.name} --project=${var.project} --zone=${var.zone} --tunnel-through-iap"
}

output "serial_port_command" {
  value = "gcloud compute instances get-serial-port-output ${google_compute_instance.kumo.name} --project=${var.project} --zone=${var.zone}"
}

output "data_disk_name" {
  value = var.data_disk_size_gb > 0 ? google_compute_disk.data[0].name : ""
}
