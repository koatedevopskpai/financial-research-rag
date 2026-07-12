# ── Terraform variables ──────────────────────────────────────────────────────
# These are the tunable parameters for the Azure infrastructure.
# Override via terraform.tfvars or environment variables (TF_VAR_*).

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "UK South"
}

variable "tags" {
  description = "Resource tags for cost tracking and organisation"
  type        = map(string)
  default = {
    project     = "financial-research-rag"
    managed_by  = "terraform"
    environment = "dev"
  }
}

variable "openai_chat_deployment_name" {
  description = "Name of the chat model deployment in Azure OpenAI"
  type        = string
  default     = "gpt-4o"
}

variable "openai_chat_model_name" {
  description = "Azure OpenAI chat model name"
  type        = string
  default     = "gpt-4o"
}

variable "openai_chat_model_version" {
  description = "Version of the chat model to deploy"
  type        = string
  default     = "2024-11-20"
}

variable "openai_chat_capacity" {
  description = "Provisioned throughput units for the chat model"
  type        = number
  default     = 10
}

variable "openai_embedding_deployment_name" {
  description = "Name of the embedding model deployment"
  type        = string
  default     = "text-embedding-ada-002"
}

variable "openai_embedding_model_name" {
  description = "Azure OpenAI embedding model name"
  type        = string
  default     = "text-embedding-ada-002"
}

variable "openai_embedding_model_version" {
  description = "Version of the embedding model"
  type        = string
  default     = "2"
}

variable "openai_embedding_capacity" {
  description = "Provisioned throughput units for the embedding model"
  type        = number
  default     = 10
}

variable "search_sku" {
  description = "Azure AI Search SKU (standard required for vector search)"
  type        = string
  default     = "standard"
}

variable "search_semantic_sku" {
  description = "Semantic ranker SKU (free for standard tier)"
  type        = string
  default     = "free"
}

variable "aks_node_count" {
  description = "Number of nodes in the AKS default node pool"
  type        = number
  default     = 2
}

variable "aks_vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_D2s_v3"
}
