# Terraform configuration for SignVerse infrastructure
# Provisions a production-ready AWS environment for AI & Dashboard services

terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  # Remote persistence for state - Highly recommended for team-based IaC
  backend "s3" {
    bucket = "signverse-tf-state"
    key    = "terraform.tfstate"
    region = "us-west-2"
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
}

# EKS Cluster: Managed Kubernetes for SignVerse Orchestration
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "signverse-cluster"
  cluster_version = "1.27"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    # General Purpose Nodes: Frontend, API Controllers, Caching
    general = {
      desired_size = 3
      min_size     = 1
      max_size     = 5

      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "general"
      }

      tags = {
        Environment = "production"
        Project     = "SignVerse"
      }
    }

    # GPU Enabled Nodes: Critical for High-Performance Pose Estimation Inference
    gpu = {
      desired_size = 1
      min_size     = 0
      max_size     = 2

      instance_types = ["g4dn.xlarge"] # NVIDIA T4 Tensor Core
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "gpu"
      }

      taints = [{
        key    = "dedicated"
        value  = "gpu"
        effect = "NO_SCHEDULE" # Ensure only GPU tasks land here
      }]

      tags = {
        Environment = "production"
        Project     = "SignVerse"
      }
    }
  }

  tags = {
    Environment = "production"
    Project     = "SignVerse"
  }
}

# VPC Module: Network Isolation for the SignVerse Ecosystem
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "signverse-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  tags = {
    Environment = "production"
    Project     = "SignVerse"
  }
}

# ECR Repositories: Managed Container Registry for SignVerse CI/CD
resource "aws_ecr_repository" "backend" {
  name                 = "signverse-backend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = "production"
    Project     = "SignVerse"
  }
}

resource "aws_ecr_repository" "frontend" {
  name                 = "signverse-frontend"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = "production"
    Project     = "SignVerse"
  }
}

# RDS PostgreSQL: Longitudinal Metadata & Application Database
module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "signverse-db"

  engine               = "postgres"
  engine_version       = "15"
  instance_class       = "db.t3.micro"
  allocated_storage    = 20
  storage_encrypted    = true
  skip_final_snapshot  = true

  db_name  = "signverse"
  username = var.db_username
  password = var.db_password
  port     = "5432"

  vpc_security_group_ids = [module.eks.cluster_primary_security_group_id]
  subnet_ids             = module.vpc.private_subnets

  maintenance_window      = "Mon:00:00-Mon:03:00"
  backup_window           = "03:00-06:00"
  backup_retention_period = 7
  multi_az                = false

  tags = {
    Environment = "production"
    Project     = "SignVerse"
  }
}

# S3 Bucket: Large-Scale Blob Storage for Video and Simulation Data
resource "aws_s3_bucket" "data" {
  bucket = "signverse-data-${var.environment}"

  tags = {
    Environment = var.environment
    Project     = "SignVerse"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
