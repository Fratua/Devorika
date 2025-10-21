"""
DevOps and Cloud Infrastructure Tools for Devorika
Supports Docker, Kubernetes, AWS, Terraform, and CI/CD pipelines.
"""

import os
import json
import subprocess
from typing import Dict, Any, List, Optional
from .base import Tool


class DockerTool(Tool):
    """
    Docker container management and operations.
    """

    name = "docker_operations"
    description = "Manage Docker containers, images, and compose files"

    def execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Execute Docker operations.

        Args:
            action: Action to perform (build, run, ps, stop, logs, compose)
            **params: Action-specific parameters

        Returns:
            Dict with operation results
        """
        try:
            if action == "build":
                return self._docker_build(params)
            elif action == "run":
                return self._docker_run(params)
            elif action == "ps":
                return self._docker_ps()
            elif action == "stop":
                return self._docker_stop(params.get('container_id'))
            elif action == "logs":
                return self._docker_logs(params.get('container_id'))
            elif action == "compose":
                return self._docker_compose(params)
            elif action == "create_dockerfile":
                return self._create_dockerfile(params)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Docker operation failed: {str(e)}"}

    def _docker_build(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build Docker image."""
        image_name = params.get('image_name', 'my-app')
        dockerfile = params.get('dockerfile', 'Dockerfile')
        context = params.get('context', '.')

        cmd = ["docker", "build", "-t", image_name, "-f", dockerfile, context]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'build',
                    'image': image_name,
                    'output': result.stdout
                }
            else:
                return {"error": f"Docker build failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "Docker not installed or not in PATH"}
        except subprocess.TimeoutExpired:
            return {"error": "Docker build timed out"}

    def _docker_run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Docker container."""
        image = params.get('image')
        if not image:
            return {"error": "Image name required"}

        cmd = ["docker", "run", "-d"]

        # Add port mappings
        if 'ports' in params:
            for port_mapping in params['ports']:
                cmd.extend(["-p", port_mapping])

        # Add environment variables
        if 'env' in params:
            for key, value in params['env'].items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add volume mounts
        if 'volumes' in params:
            for volume in params['volumes']:
                cmd.extend(["-v", volume])

        # Add name
        if 'name' in params:
            cmd.extend(["--name", params['name']])

        cmd.append(image)

        # Add command
        if 'command' in params:
            cmd.extend(params['command'].split())

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                container_id = result.stdout.strip()
                return {
                    'success': True,
                    'action': 'run',
                    'container_id': container_id,
                    'image': image
                }
            else:
                return {"error": f"Docker run failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "Docker not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Docker run timed out"}

    def _docker_ps(self) -> Dict[str, Any]:
        """List running containers."""
        cmd = ["docker", "ps", "--format", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            containers.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

                return {
                    'success': True,
                    'action': 'ps',
                    'containers': containers,
                    'count': len(containers)
                }
            else:
                return {"error": f"Docker ps failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "Docker not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Docker ps timed out"}

    def _docker_stop(self, container_id: str) -> Dict[str, Any]:
        """Stop container."""
        if not container_id:
            return {"error": "Container ID required"}

        cmd = ["docker", "stop", container_id]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'stop',
                    'container_id': container_id
                }
            else:
                return {"error": f"Docker stop failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Docker stop timed out"}

    def _docker_logs(self, container_id: str) -> Dict[str, Any]:
        """Get container logs."""
        if not container_id:
            return {"error": "Container ID required"}

        cmd = ["docker", "logs", "--tail", "100", container_id]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return {
                'success': True,
                'action': 'logs',
                'container_id': container_id,
                'logs': result.stdout
            }

        except subprocess.TimeoutExpired:
            return {"error": "Docker logs timed out"}

    def _docker_compose(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute docker-compose commands."""
        compose_file = params.get('compose_file', 'docker-compose.yml')
        action = params.get('compose_action', 'up')

        cmd = ["docker-compose", "-f", compose_file, action]

        if action == "up":
            cmd.append("-d")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': f'compose_{action}',
                    'compose_file': compose_file,
                    'output': result.stdout
                }
            else:
                return {"error": f"Docker compose failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "docker-compose not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Docker compose timed out"}

    def _create_dockerfile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Dockerfile for project."""
        language = params.get('language', 'python')
        output_file = params.get('output_file', 'Dockerfile')

        if language == 'python':
            dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
"""
        elif language == 'node':
            dockerfile_content = """FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "index.js"]
"""
        elif language == 'go':
            dockerfile_content = """FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN go build -o main .

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/main .

EXPOSE 8080

CMD ["./main"]
"""
        else:
            return {"error": f"Unsupported language: {language}"}

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        return {
            'success': True,
            'action': 'create_dockerfile',
            'file': output_file,
            'language': language
        }


class KubernetesTool(Tool):
    """
    Kubernetes cluster management and operations.
    """

    name = "kubernetes_operations"
    description = "Manage Kubernetes deployments, services, and pods"

    def execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Execute Kubernetes operations.

        Args:
            action: Action (get, apply, delete, logs, create_manifest)
            **params: Action-specific parameters

        Returns:
            Dict with operation results
        """
        try:
            if action == "get":
                return self._kubectl_get(params)
            elif action == "apply":
                return self._kubectl_apply(params)
            elif action == "delete":
                return self._kubectl_delete(params)
            elif action == "logs":
                return self._kubectl_logs(params)
            elif action == "create_manifest":
                return self._create_k8s_manifest(params)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Kubernetes operation failed: {str(e)}"}

    def _kubectl_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Kubernetes resources."""
        resource = params.get('resource', 'pods')
        namespace = params.get('namespace', 'default')

        cmd = ["kubectl", "get", resource, "-n", namespace, "-o", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                resources = json.loads(result.stdout)
                return {
                    'success': True,
                    'action': 'get',
                    'resource': resource,
                    'namespace': namespace,
                    'items': resources.get('items', [])
                }
            else:
                return {"error": f"kubectl get failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "kubectl not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "kubectl get timed out"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse kubectl output"}

    def _kubectl_apply(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Kubernetes manifest."""
        manifest_file = params.get('manifest_file')
        if not manifest_file:
            return {"error": "Manifest file required"}

        cmd = ["kubectl", "apply", "-f", manifest_file]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'apply',
                    'manifest': manifest_file,
                    'output': result.stdout
                }
            else:
                return {"error": f"kubectl apply failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "kubectl not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "kubectl apply timed out"}

    def _kubectl_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete Kubernetes resource."""
        resource = params.get('resource')
        name = params.get('name')

        if not resource or not name:
            return {"error": "Both resource and name required"}

        namespace = params.get('namespace', 'default')
        cmd = ["kubectl", "delete", resource, name, "-n", namespace]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'delete',
                    'resource': resource,
                    'name': name
                }
            else:
                return {"error": f"kubectl delete failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "kubectl delete timed out"}

    def _kubectl_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get pod logs."""
        pod_name = params.get('pod_name')
        if not pod_name:
            return {"error": "Pod name required"}

        namespace = params.get('namespace', 'default')
        cmd = ["kubectl", "logs", pod_name, "-n", namespace, "--tail=100"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return {
                'success': True,
                'action': 'logs',
                'pod': pod_name,
                'logs': result.stdout
            }

        except subprocess.TimeoutExpired:
            return {"error": "kubectl logs timed out"}

    def _create_k8s_manifest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Kubernetes manifest files."""
        app_name = params.get('app_name', 'my-app')
        image = params.get('image', 'my-app:latest')
        port = params.get('port', 8000)
        replicas = params.get('replicas', 3)
        output_file = params.get('output_file', 'k8s-manifest.yaml')

        manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  selector:
    app: {app_name}
  ports:
  - protocol: TCP
    port: 80
    targetPort: {port}
  type: LoadBalancer
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(manifest)

        return {
            'success': True,
            'action': 'create_manifest',
            'file': output_file,
            'app_name': app_name
        }


class TerraformTool(Tool):
    """
    Infrastructure as Code with Terraform.
    """

    name = "terraform_operations"
    description = "Manage infrastructure with Terraform (init, plan, apply, destroy)"

    def execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Execute Terraform operations.

        Args:
            action: Action (init, plan, apply, destroy, create_config)
            **params: Action-specific parameters

        Returns:
            Dict with operation results
        """
        try:
            working_dir = params.get('working_dir', '.')

            if action == "init":
                return self._terraform_init(working_dir)
            elif action == "plan":
                return self._terraform_plan(working_dir)
            elif action == "apply":
                return self._terraform_apply(working_dir, params.get('auto_approve', False))
            elif action == "destroy":
                return self._terraform_destroy(working_dir, params.get('auto_approve', False))
            elif action == "create_config":
                return self._create_terraform_config(params)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": f"Terraform operation failed: {str(e)}"}

    def _terraform_init(self, working_dir: str) -> Dict[str, Any]:
        """Initialize Terraform."""
        cmd = ["terraform", "init"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=working_dir, timeout=120)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'init',
                    'output': result.stdout
                }
            else:
                return {"error": f"Terraform init failed: {result.stderr}"}

        except FileNotFoundError:
            return {"error": "Terraform not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "Terraform init timed out"}

    def _terraform_plan(self, working_dir: str) -> Dict[str, Any]:
        """Run Terraform plan."""
        cmd = ["terraform", "plan", "-out=tfplan"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=working_dir, timeout=120)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'plan',
                    'output': result.stdout,
                    'plan_file': 'tfplan'
                }
            else:
                return {"error": f"Terraform plan failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Terraform plan timed out"}

    def _terraform_apply(self, working_dir: str, auto_approve: bool) -> Dict[str, Any]:
        """Apply Terraform changes."""
        cmd = ["terraform", "apply"]
        if auto_approve:
            cmd.append("-auto-approve")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=working_dir, timeout=300)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'apply',
                    'output': result.stdout
                }
            else:
                return {"error": f"Terraform apply failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Terraform apply timed out"}

    def _terraform_destroy(self, working_dir: str, auto_approve: bool) -> Dict[str, Any]:
        """Destroy Terraform-managed infrastructure."""
        cmd = ["terraform", "destroy"]
        if auto_approve:
            cmd.append("-auto-approve")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=working_dir, timeout=300)

            if result.returncode == 0:
                return {
                    'success': True,
                    'action': 'destroy',
                    'output': result.stdout
                }
            else:
                return {"error": f"Terraform destroy failed: {result.stderr}"}

        except subprocess.TimeoutExpired:
            return {"error": "Terraform destroy timed out"}

    def _create_terraform_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Terraform configuration files."""
        provider = params.get('provider', 'aws')
        output_file = params.get('output_file', 'main.tf')

        if provider == 'aws':
            config = """terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

resource "aws_instance" "app_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "DevorikaAppServer"
  }
}

output "instance_ip" {
  value = aws_instance.app_server.public_ip
}
"""
        elif provider == 'gcp':
            config = """terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "GCP Project ID"
}

variable "region" {
  description = "GCP region"
  default     = "us-central1"
}

resource "google_compute_instance" "app_server" {
  name         = "devorika-app-server"
  machine_type = "e2-micro"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
  }
}
"""
        else:
            return {"error": f"Unsupported provider: {provider}"}

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(config)

        return {
            'success': True,
            'action': 'create_config',
            'file': output_file,
            'provider': provider
        }


class CICDTool(Tool):
    """
    CI/CD pipeline management (GitHub Actions, GitLab CI).
    """

    name = "cicd_operations"
    description = "Create and manage CI/CD pipelines"

    def execute(self, platform: str, action: str, **params) -> Dict[str, Any]:
        """
        Manage CI/CD pipelines.

        Args:
            platform: Platform (github, gitlab, jenkins)
            action: Action (create_pipeline, validate)
            **params: Platform-specific parameters

        Returns:
            Dict with operation results
        """
        try:
            if platform == "github":
                return self._github_actions(action, params)
            elif platform == "gitlab":
                return self._gitlab_ci(action, params)
            elif platform == "jenkins":
                return self._jenkins(action, params)
            else:
                return {"error": f"Unsupported platform: {platform}"}

        except Exception as e:
            return {"error": f"CI/CD operation failed: {str(e)}"}

    def _github_actions(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create GitHub Actions workflow."""
        if action == "create_pipeline":
            language = params.get('language', 'python')
            output_dir = params.get('output_dir', '.github/workflows')
            output_file = os.path.join(output_dir, 'ci.yml')

            os.makedirs(output_dir, exist_ok=True)

            if language == 'python':
                workflow = """name: Python CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Install linting tools
      run: pip install pylint black
    - name: Run linters
      run: |
        black --check .
        pylint **/*.py
"""
            elif language == 'node':
                workflow = """name: Node.js CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16.x, 18.x, 20.x]

    steps:
    - uses: actions/checkout@v3
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
    - run: npm ci
    - run: npm run build --if-present
    - run: npm test
"""
            else:
                return {"error": f"Unsupported language: {language}"}

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(workflow)

            return {
                'success': True,
                'platform': 'github',
                'file': output_file,
                'language': language
            }

        return {"error": f"Unknown action: {action}"}

    def _gitlab_ci(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create GitLab CI configuration."""
        if action == "create_pipeline":
            language = params.get('language', 'python')
            output_file = params.get('output_file', '.gitlab-ci.yml')

            if language == 'python':
                config = """stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=. --cov-report=term
  coverage: '/TOTAL.*\s+(\d+%)$/'

lint:
  stage: test
  image: python:3.11
  script:
    - pip install pylint black
    - black --check .
    - pylint **/*.py

build:
  stage: build
  script:
    - echo "Building application..."
  only:
    - main
"""
            else:
                return {"error": f"Unsupported language: {language}"}

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(config)

            return {
                'success': True,
                'platform': 'gitlab',
                'file': output_file,
                'language': language
            }

        return {"error": f"Unknown action: {action}"}

    def _jenkins(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create Jenkinsfile."""
        if action == "create_pipeline":
            output_file = params.get('output_file', 'Jenkinsfile')

            jenkinsfile = """pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Test') {
            steps {
                sh 'pytest --cov=. --cov-report=xml'
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying application...'
            }
        }
    }

    post {
        always {
            junit '**/test-results/*.xml'
            publishHTML target: [
                reportDir: 'coverage',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ]
        }
    }
}
"""

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(jenkinsfile)

            return {
                'success': True,
                'platform': 'jenkins',
                'file': output_file
            }

        return {"error": f"Unknown action: {action}"}
