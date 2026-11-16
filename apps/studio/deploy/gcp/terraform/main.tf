# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "= 7.39.0"
    }
  }
}

locals {
  instance_name   = var.instance_name != "" ? var.instance_name : var.deployment_name
  region          = join("-", slice(split("-", var.zone), 0, 2))
  dataplane_root  = var.dataplane_root != "" ? var.dataplane_root : "gs://kumo-${var.deployment_name}/kumo/dataplane"
  gcs_bucket_name = split("/", replace(local.dataplane_root, "gs://", ""))[0]
  dataplane_prefix = trim(replace(
    local.dataplane_root,
    "gs://${local.gcs_bucket_name}",
    "",
  ), "/")

  # SA name: GCP limit is 6-30 chars; prefix "kumo-" = 5 chars, so the deployment name is truncated to 24.
  sa_name   = "kumo-${substr(var.deployment_name, 0, min(length(var.deployment_name), 24))}"
  sa_email  = var.service_account_email != "" ? var.service_account_email : "${local.sa_name}@${var.project}.iam.gserviceaccount.com"
  create_sa = var.service_account_email == ""

  kumo_config = join("\n", compact([
    "dataplane_root: ${local.dataplane_root}",
    "lifecycle_ready: true",
    var.enable_tls != "" ? "enable_tls: ${var.enable_tls}" : "",
    var.public_url != "" ? "public_url: ${var.public_url}" : "",
    var.allow_local_public_url != "" ? "allow_local_public_url: ${var.allow_local_public_url}" : "",
    var.disable_auth != "" ? "disable_auth: ${var.disable_auth}" : "",
    var.oidc_issuer != "" ? "oidc_issuer: ${var.oidc_issuer}" : "",
    var.oidc_client_id != "" ? "oidc_client_id: ${var.oidc_client_id}" : "",
    var.oidc_client_secret != "" ? "oidc_client_secret: ${var.oidc_client_secret}" : "",
    var.oidc_groups_claim != "" ? "oidc_groups_claim: ${var.oidc_groups_claim}" : "",
    var.direct_connect_login_method != "" ? "direct_connect_login_method: ${var.direct_connect_login_method}" : "",
    var.admin_emails != "" ? "admin_emails: ${var.admin_emails}" : "",
    var.direct_connect_admin_temp_password != "" ? "direct_connect_admin_temp_password: ${var.direct_connect_admin_temp_password}" : "",
  ]))
}

# --- Service account ---

resource "google_service_account" "kumo" {
  count        = local.create_sa ? 1 : 0
  account_id   = local.sa_name
  display_name = "Kumo singlehost (${var.deployment_name})"
  project      = var.project
}

# --- Project-level IAM roles ---

resource "google_project_iam_member" "secret_accessor" {
  project    = var.project
  role       = "roles/secretmanager.secretAccessor"
  member     = "serviceAccount:${local.sa_email}"
  depends_on = [google_service_account.kumo]
}

resource "google_project_iam_member" "log_writer" {
  project    = var.project
  role       = "roles/logging.logWriter"
  member     = "serviceAccount:${local.sa_email}"
  depends_on = [google_service_account.kumo]
}

resource "google_project_iam_member" "metric_writer" {
  project    = var.project
  role       = "roles/monitoring.metricWriter"
  member     = "serviceAccount:${local.sa_email}"
  depends_on = [google_service_account.kumo]
}

# Self-impersonation: allows the SA to call signBlob on itself, which is
# required by the GCS presigned URL code path (impl_gcs.py _get_signing_credentials).
# Scoped to the SA resource (not project-wide) to follow least privilege.
resource "google_service_account_iam_member" "self_token_creator" {
  count              = local.create_sa ? 1 : 0
  service_account_id = google_service_account.kumo[0].name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${local.sa_email}"
}

# --- GCS dataplane bucket ---

# NOTE(trivy): GCP-0066 — no customer-managed KMS keys provisioned here.
# GCP-0078 — versioning is a per-deployment owner decision (bucket mixes
# regenerable exports with uploads/model artifacts; versioning parquet is costly).
#trivy:ignore:AVD-GCP-0066
#trivy:ignore:AVD-GCP-0078
resource "google_storage_bucket" "dataplane" {
  name                        = local.gcs_bucket_name
  project                     = var.project
  location                    = local.region
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    condition {
      age = var.temporary_object_retention_days
      matches_prefix = [
        "${local.dataplane_prefix != "" ? "${local.dataplane_prefix}/" : ""}tmp_payloads/",
        "${local.dataplane_prefix != "" ? "${local.dataplane_prefix}/" : ""}temporary-uploaded-datasets/",
        "${local.dataplane_prefix != "" ? "${local.dataplane_prefix}/" : ""}tmp_archives/",
      ]
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket_iam_member" "dataplane_object_admin" {
  bucket     = google_storage_bucket.dataplane.name
  role       = "roles/storage.objectAdmin"
  member     = "serviceAccount:${local.sa_email}"
  depends_on = [google_service_account.kumo]
}

# --- Subnet (optional, created when create_subnet = true) ---
# If the customer brings their own subnet, they are responsible for enabling
# Private Google Access on it before applying this module.

resource "google_compute_subnetwork" "kumo" {
  count                    = var.create_subnet ? 1 : 0
  name                     = "kumo-${var.deployment_name}"
  project                  = var.project
  region                   = local.region
  network                  = var.network
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = var.enable_private_google_access
}

locals {
  # Resolved subnet self-link: created subnet, explicit subnet, or null (default network).
  resolved_subnet = (
    var.create_subnet ? google_compute_subnetwork.kumo[0].self_link :
    var.subnet != "" ? var.subnet :
    null
  )
}

# --- IAP firewall rule (port 22 from IAP range) ---

resource "google_compute_firewall" "iap_ssh" {
  count   = var.create_iap_firewall_rule ? 1 : 0
  name    = "kumo-${var.deployment_name}-iap-ssh"
  project = var.project
  network = var.network

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges           = ["35.235.240.0/20"]
  target_service_accounts = [local.sa_email]

  depends_on = [google_service_account.kumo]
}

# --- Optional persistent data disk ---

resource "google_compute_disk" "data" {
  count   = var.data_disk_size_gb > 0 ? 1 : 0
  name    = "${local.instance_name}-data"
  project = var.project
  zone    = var.zone
  type    = var.data_disk_type
  size    = var.data_disk_size_gb
}

# --- Local SSDs (one scratch disk block per count) ---

# NOTE(trivy): Shielded VM needs a UEFI-compatible image; var.image_name is
# customer-supplied — enabling is an owner decision.
#trivy:ignore:AVD-GCP-0041
#trivy:ignore:AVD-GCP-0045
#trivy:ignore:AVD-GCP-0067
resource "google_compute_instance" "kumo" {
  name         = local.instance_name
  project      = var.project
  zone         = var.zone
  machine_type = var.machine_type

  # NOTE(trivy): no customer-managed disk-encryption keys provisioned here.
  #trivy:ignore:AVD-GCP-0033
  boot_disk {
    initialize_params {
      image = "projects/${var.project}/global/images/${var.image_name}"
      size  = var.boot_disk_size_gb
      type  = var.boot_disk_type
    }
  }

  dynamic "scratch_disk" {
    for_each = range(var.local_ssd_count)
    content {
      interface = var.local_ssd_interface
    }
  }

  dynamic "attached_disk" {
    for_each = var.data_disk_size_gb > 0 ? [1] : []
    content {
      source      = google_compute_disk.data[0].self_link
      device_name = "kumo-data"
      mode        = "READ_WRITE"
    }
  }

  guest_accelerator {
    type  = var.accelerator_type
    count = var.accelerator_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    network    = local.resolved_subnet == null ? var.network : null
    subnetwork = local.resolved_subnet

    dynamic "access_config" {
      for_each = var.enable_public_ip ? [1] : []
      content {
        # ephemeral public IP
      }
    }
  }

  service_account {
    email  = local.sa_email
    scopes = ["cloud-platform"]
  }

  # NOTE(trivy): project-SSH-key blocking and OS Login change operator access
  # to customer VMs — owner decisions.
  #trivy:ignore:AVD-GCP-0030
  #trivy:ignore:AVD-GCP-0036
  metadata = {
    serial-port-logging-enable = "true"
    kumo-config                = local.kumo_config
  }

  depends_on = [
    google_service_account.kumo,
    google_storage_bucket_iam_member.dataplane_object_admin,
    google_compute_disk.data,
  ]
}
