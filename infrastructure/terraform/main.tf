# ── Terraform configuration for Azure infrastructure ─────────────────────────
# Provisioning order:
#   1. Resource group
#   2. Azure OpenAI (model deployments for GPT-4o + text-embedding-ada-002)
#   3. Azure AI Search (with semantic ranker)
#   4. Azure Container Registry (for Docker images)
#   5. Azure Kubernetes Service (for app deployment)
#
# Design decisions:
#   - Separate dev/staging/prod via workspaces or tfvars files
#   - RBAC via Managed Identity (no API keys in config)
#   - Azure AI Search is Standard tier (free tier doesn't support vector search)

terraform {
  required_version = ">= 1.7"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  # In production, use a remote backend (e.g. Azure Storage):
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "ragtfstate"
  #   container_name       = "tfstate"
  #   key                  = "financial-research-rag.tfstate"
  # }
}

provider "azurerm" {
  features {}
}

# ── Resource Group ───────────────────────────────────────────────────────────
resource "azurerm_resource_group" "main" {
  name     = "rg-financial-research-rag-${var.environment}"
  location = var.location
  tags     = var.tags
}

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
resource "azurerm_cognitive_account" "openai" {
  name                = "cog-financial-rag-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"
  custom_subdomain_name = "cog-financial-rag-${var.environment}"
  tags                = var.tags
}

# Deploy the chat model (e.g. GPT-4o)
resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.openai_chat_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.openai_chat_model_name
    version = var.openai_chat_model_version
  }
  scale {
    type     = "Standard"
    capacity = var.openai_chat_capacity
  }
}

# Deploy the embedding model (e.g. text-embedding-ada-002)
resource "azurerm_cognitive_deployment" "embedding" {
  name                 = var.openai_embedding_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.openai_embedding_model_name
    version = var.openai_embedding_model_version
  }
  scale {
    type     = "Standard"
    capacity = var.openai_embedding_capacity
  }
}

# ── Azure AI Search ──────────────────────────────────────────────────────────
# Standard tier is required for vector search + semantic ranker.
resource "azurerm_search_service" "search" {
  name                = "srch-financial-rag-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.search_sku
  semantic_search_sku = var.search_semantic_sku
  tags                = var.tags
}

# ── Azure Container Registry ─────────────────────────────────────────────────
resource "azurerm_container_registry" "acr" {
  name                = "acrfinancialrag${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = false
  tags                = var.tags
}

# ── Azure Kubernetes Service ─────────────────────────────────────────────────
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "aks-financial-rag-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "finrag${var.environment}"

  default_node_pool {
    name       = "default"
    node_count = var.aks_node_count
    vm_size    = var.aks_vm_size
  }

  identity {
    type = "SystemAssigned"
  }

  # ACR integration — allows AKS to pull images without credentials
  role_based_access_control_enabled = true
  azure_policy_enabled              = true

  tags = var.tags
}

# Grant AKS pull access to ACR
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id         = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name = "AcrPull"
  scope                = azurerm_container_registry.acr.id
}
