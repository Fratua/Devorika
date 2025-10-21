"""
Security Tools for Devorika
Comprehensive security scanning, vulnerability detection, and compliance checking.
"""

import os
import re
import json
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base import Tool


class VulnerabilityScannerTool(Tool):
    """
    Scan dependencies for known vulnerabilities using safety and pip-audit.
    """

    name = "vulnerability_scanner"
    description = "Scan project dependencies for security vulnerabilities"

    def execute(self, project_dir: str = ".", scanner: str = "safety",
                requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """
        Scan for vulnerabilities.

        Args:
            project_dir: Project directory
            scanner: Scanner to use (safety, pip-audit, bandit)
            requirements_file: Requirements file path

        Returns:
            Dict with vulnerability scan results
        """
        try:
            if scanner == "safety":
                return self._run_safety(project_dir, requirements_file)
            elif scanner == "pip-audit":
                return self._run_pip_audit(project_dir)
            elif scanner == "bandit":
                return self._run_bandit(project_dir)
            else:
                return {"error": f"Unknown scanner: {scanner}"}

        except Exception as e:
            return {"error": f"Vulnerability scan failed: {str(e)}"}

    def _run_safety(self, project_dir: str, requirements_file: str) -> Dict[str, Any]:
        """Run Safety scanner."""
        req_path = os.path.join(project_dir, requirements_file)

        if not os.path.exists(req_path):
            return {"error": f"Requirements file not found: {req_path}"}

        cmd = ["safety", "check", "--file", req_path, "--json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Safety returns non-zero on vulnerabilities
            try:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
            except json.JSONDecodeError:
                vulnerabilities = []

            return {
                'success': True,
                'scanner': 'safety',
                'vulnerabilities': vulnerabilities,
                'count': len(vulnerabilities),
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "Safety not installed. Install with: pip install safety"}
        except subprocess.TimeoutExpired:
            return {"error": "Safety scan timed out"}

    def _run_pip_audit(self, project_dir: str) -> Dict[str, Any]:
        """Run pip-audit scanner."""
        cmd = ["pip-audit", "--format", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=project_dir, timeout=120)

            try:
                scan_results = json.loads(result.stdout) if result.stdout else {}
                vulnerabilities = scan_results.get('dependencies', [])
            except json.JSONDecodeError:
                vulnerabilities = []

            return {
                'success': True,
                'scanner': 'pip-audit',
                'vulnerabilities': vulnerabilities,
                'count': len(vulnerabilities),
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "pip-audit not installed. Install with: pip install pip-audit"}
        except subprocess.TimeoutExpired:
            return {"error": "pip-audit scan timed out"}

    def _run_bandit(self, project_dir: str) -> Dict[str, Any]:
        """Run Bandit security linter."""
        cmd = ["bandit", "-r", project_dir, "-f", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            try:
                scan_results = json.loads(result.stdout) if result.stdout else {}
                issues = scan_results.get('results', [])
            except json.JSONDecodeError:
                issues = []

            # Categorize by severity
            severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for issue in issues:
                severity = issue.get('issue_severity', 'LOW')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            return {
                'success': True,
                'scanner': 'bandit',
                'issues': issues,
                'count': len(issues),
                'severity_counts': severity_counts,
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "Bandit not installed. Install with: pip install bandit"}
        except subprocess.TimeoutExpired:
            return {"error": "Bandit scan timed out"}


class SecretDetectionTool(Tool):
    """
    Detect hardcoded secrets, API keys, and credentials in code.
    """

    name = "secret_detection"
    description = "Detect hardcoded secrets, passwords, and API keys in codebase"

    def execute(self, project_dir: str = ".", tool: str = "detect-secrets") -> Dict[str, Any]:
        """
        Detect secrets in codebase.

        Args:
            project_dir: Project directory to scan
            tool: Tool to use (detect-secrets, trufflehog, custom)

        Returns:
            Dict with detected secrets
        """
        try:
            if tool == "detect-secrets":
                return self._run_detect_secrets(project_dir)
            elif tool == "custom":
                return self._custom_secret_scan(project_dir)
            else:
                return {"error": f"Unknown tool: {tool}"}

        except Exception as e:
            return {"error": f"Secret detection failed: {str(e)}"}

    def _run_detect_secrets(self, project_dir: str) -> Dict[str, Any]:
        """Run detect-secrets scanner."""
        cmd = ["detect-secrets", "scan", project_dir]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            try:
                scan_results = json.loads(result.stdout) if result.stdout else {}
                secrets = scan_results.get('results', {})
            except json.JSONDecodeError:
                secrets = {}

            # Count total secrets
            total_secrets = sum(len(file_secrets) for file_secrets in secrets.values())

            return {
                'success': True,
                'tool': 'detect-secrets',
                'secrets': secrets,
                'count': total_secrets,
                'files_with_secrets': len(secrets),
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "detect-secrets not installed. Install with: pip install detect-secrets"}
        except subprocess.TimeoutExpired:
            return {"error": "Secret detection timed out"}

    def _custom_secret_scan(self, project_dir: str) -> Dict[str, Any]:
        """Custom secret detection using regex patterns."""
        secret_patterns = {
            'AWS Access Key': r'AKIA[0-9A-Z]{16}',
            'API Key': r'(?i)api[_-]?key[\s]*[:=][\s]*["\']?([a-zA-Z0-9_-]{20,})',
            'Password': r'(?i)password[\s]*[:=][\s]*["\']([^"\']+)',
            'Private Key': r'-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY',
            'JWT Token': r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
            'Generic Secret': r'(?i)secret[\s]*[:=][\s]*["\']([^"\']+)',
        }

        findings = []

        for root, dirs, files in os.walk(project_dir):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', 'env', 'node_modules'}]

            for file in files:
                if file.endswith(('.py', '.js', '.java', '.go', '.rb', '.env', '.yaml', '.yml', '.json')):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        for secret_type, pattern in secret_patterns.items():
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                # Get line number
                                line_num = content[:match.start()].count('\n') + 1

                                findings.append({
                                    'type': secret_type,
                                    'file': file_path,
                                    'line': line_num,
                                    'match': match.group(0)[:50] + '...' if len(match.group(0)) > 50 else match.group(0)
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return {
            'success': True,
            'tool': 'custom',
            'findings': findings,
            'count': len(findings),
            'project_dir': project_dir
        }


class SASTTool(Tool):
    """
    Static Application Security Testing using various tools.
    """

    name = "sast_scan"
    description = "Perform static application security testing"

    def execute(self, project_dir: str = ".", language: str = "python",
                tool: str = "auto") -> Dict[str, Any]:
        """
        Perform SAST scan.

        Args:
            project_dir: Project directory
            language: Programming language
            tool: SAST tool to use

        Returns:
            Dict with SAST results
        """
        try:
            if language == "python":
                return self._python_sast(project_dir)
            elif language == "javascript":
                return self._javascript_sast(project_dir)
            else:
                return {"error": f"Unsupported language: {language}"}

        except Exception as e:
            return {"error": f"SAST scan failed: {str(e)}"}

    def _python_sast(self, project_dir: str) -> Dict[str, Any]:
        """Python SAST using Bandit and custom checks."""
        # Run Bandit
        cmd = ["bandit", "-r", project_dir, "-f", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            try:
                scan_results = json.loads(result.stdout) if result.stdout else {}
            except json.JSONDecodeError:
                scan_results = {}

            # Custom security checks
            custom_issues = self._custom_python_security_checks(project_dir)

            return {
                'success': True,
                'language': 'python',
                'bandit_results': scan_results.get('results', []),
                'custom_checks': custom_issues,
                'total_issues': len(scan_results.get('results', [])) + len(custom_issues),
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "Bandit not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "SAST scan timed out"}

    def _custom_python_security_checks(self, project_dir: str) -> List[Dict[str, Any]]:
        """Custom Python security checks."""
        issues = []

        dangerous_patterns = {
            'eval() usage': r'\beval\s*\(',
            'exec() usage': r'\bexec\s*\(',
            'pickle usage': r'import\s+pickle|from\s+pickle',
            'SQL injection risk': r'execute\s*\(\s*["\'].*%s.*["\']',
            'Command injection': r'os\.system\s*\(|subprocess\.call\s*\(',
            'Hardcoded password': r'password\s*=\s*["\'][^"\']+["\']',
        }

        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', 'env'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for check_name, pattern in dangerous_patterns.items():
                            if re.search(pattern, content):
                                line_num = content[:re.search(pattern, content).start()].count('\n') + 1
                                issues.append({
                                    'check': check_name,
                                    'file': file_path,
                                    'line': line_num,
                                    'severity': 'HIGH'
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _javascript_sast(self, project_dir: str) -> Dict[str, Any]:
        """JavaScript SAST using ESLint security plugin."""
        cmd = ["eslint", "--format", "json", project_dir]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            try:
                issues = json.loads(result.stdout) if result.stdout else []
            except json.JSONDecodeError:
                issues = []

            return {
                'success': True,
                'language': 'javascript',
                'issues': issues,
                'count': len(issues),
                'project_dir': project_dir
            }

        except FileNotFoundError:
            return {"error": "ESLint not installed"}
        except subprocess.TimeoutExpired:
            return {"error": "JavaScript SAST timed out"}


class ComplianceCheckerTool(Tool):
    """
    Check code compliance with security standards (OWASP, PCI-DSS, etc.).
    """

    name = "compliance_checker"
    description = "Check compliance with security standards and best practices"

    def execute(self, project_dir: str = ".", standard: str = "owasp") -> Dict[str, Any]:
        """
        Check compliance with security standards.

        Args:
            project_dir: Project directory
            standard: Standard to check (owasp, pci-dss, gdpr)

        Returns:
            Dict with compliance results
        """
        try:
            if standard == "owasp":
                return self._owasp_compliance(project_dir)
            elif standard == "pci-dss":
                return self._pci_dss_compliance(project_dir)
            elif standard == "gdpr":
                return self._gdpr_compliance(project_dir)
            else:
                return {"error": f"Unknown standard: {standard}"}

        except Exception as e:
            return {"error": f"Compliance check failed: {str(e)}"}

    def _owasp_compliance(self, project_dir: str) -> Dict[str, Any]:
        """Check OWASP Top 10 compliance."""
        checks = {
            'A01: Broken Access Control': self._check_access_control(project_dir),
            'A02: Cryptographic Failures': self._check_cryptography(project_dir),
            'A03: Injection': self._check_injection(project_dir),
            'A05: Security Misconfiguration': self._check_security_config(project_dir),
            'A07: Authentication Failures': self._check_authentication(project_dir),
        }

        total_issues = sum(len(issues) for issues in checks.values())
        passed_checks = sum(1 for issues in checks.values() if len(issues) == 0)

        return {
            'success': True,
            'standard': 'OWASP Top 10',
            'checks': checks,
            'total_issues': total_issues,
            'passed_checks': passed_checks,
            'total_checks': len(checks),
            'compliance_score': f"{(passed_checks / len(checks)) * 100:.1f}%"
        }

    def _check_access_control(self, project_dir: str) -> List[Dict[str, Any]]:
        """Check for access control issues."""
        issues = []
        pattern = r'@login_required|@permission_required|check_permission|authorize'

        # Look for routes/views without auth decorators
        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Find route definitions without auth
                        route_pattern = r'@app\.route\([^)]+\)\s*\ndef\s+(\w+)'
                        for match in re.finditer(route_pattern, content):
                            # Check if there's an auth decorator before this
                            func_start = match.start()
                            preceding_text = content[max(0, func_start-200):func_start]

                            if not re.search(pattern, preceding_text):
                                line_num = content[:func_start].count('\n') + 1
                                issues.append({
                                    'type': 'Missing access control',
                                    'file': file_path,
                                    'line': line_num,
                                    'function': match.group(1)
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _check_cryptography(self, project_dir: str) -> List[Dict[str, Any]]:
        """Check for cryptographic issues."""
        issues = []

        weak_crypto = {
            'MD5 usage': r'\bmd5\s*\(',
            'SHA1 usage': r'\bsha1\s*\(',
            'Weak encryption': r'DES|RC4',
            'Hardcoded key': r'(?i)key\s*=\s*["\'][^"\']{8,}["\']',
        }

        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for issue_type, pattern in weak_crypto.items():
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                issues.append({
                                    'type': issue_type,
                                    'file': file_path,
                                    'line': line_num
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _check_injection(self, project_dir: str) -> List[Dict[str, Any]]:
        """Check for injection vulnerabilities."""
        issues = []

        injection_patterns = {
            'SQL Injection': r'execute\s*\(\s*["\'].*%.*["\']|cursor\.execute\s*\(.*\+',
            'Command Injection': r'os\.system\s*\(.*\+|subprocess\.\w+\s*\(.*\+',
            'LDAP Injection': r'search\s*\(.*\+',
        }

        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for issue_type, pattern in injection_patterns.items():
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                issues.append({
                                    'type': issue_type,
                                    'file': file_path,
                                    'line': line_num
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _check_security_config(self, project_dir: str) -> List[Dict[str, Any]]:
        """Check for security misconfigurations."""
        issues = []

        # Check for debug mode in production
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if file.endswith(('.py', '.env', '.config')):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if re.search(r'DEBUG\s*=\s*True', content, re.IGNORECASE):
                            issues.append({
                                'type': 'Debug mode enabled',
                                'file': file_path,
                                'severity': 'HIGH'
                            })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _check_authentication(self, project_dir: str) -> List[Dict[str, Any]]:
        """Check for authentication issues."""
        issues = []

        auth_patterns = {
            'Weak password check': r'len\(password\)\s*[<>=]+\s*[0-7]',
            'No password hashing': r'password\s*==|password\s*=\s*request',
        }

        for root, dirs, files in os.walk(project_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for issue_type, pattern in auth_patterns.items():
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                issues.append({
                                    'type': issue_type,
                                    'file': file_path,
                                    'line': line_num
                                })

                    except (UnicodeDecodeError, PermissionError):
                        continue

        return issues

    def _pci_dss_compliance(self, project_dir: str) -> Dict[str, Any]:
        """Check PCI-DSS compliance."""
        # Simplified PCI-DSS checks
        return {
            'success': True,
            'standard': 'PCI-DSS',
            'message': 'PCI-DSS compliance requires manual review and additional tools'
        }

    def _gdpr_compliance(self, project_dir: str) -> Dict[str, Any]:
        """Check GDPR compliance."""
        # Check for data protection measures
        issues = []

        # Look for data collection without consent
        gdpr_patterns = {
            'Data collection': r'collect_user_data|store_personal_info',
            'Encryption': r'encrypt|aes|rsa',
            'Consent': r'consent|gdpr|privacy_policy',
        }

        has_encryption = False
        has_consent = False

        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if re.search(gdpr_patterns['Encryption'], content, re.IGNORECASE):
                            has_encryption = True
                        if re.search(gdpr_patterns['Consent'], content, re.IGNORECASE):
                            has_consent = True

                    except (UnicodeDecodeError, PermissionError):
                        continue

        if not has_encryption:
            issues.append({'type': 'Missing data encryption', 'severity': 'HIGH'})
        if not has_consent:
            issues.append({'type': 'Missing consent mechanism', 'severity': 'HIGH'})

        return {
            'success': True,
            'standard': 'GDPR',
            'issues': issues,
            'has_encryption': has_encryption,
            'has_consent': has_consent
        }


class LicenseCheckerTool(Tool):
    """
    Check licenses of dependencies for compliance.
    """

    name = "license_checker"
    description = "Check licenses of project dependencies"

    def execute(self, project_dir: str = ".", allowed_licenses: List[str] = None) -> Dict[str, Any]:
        """
        Check dependency licenses.

        Args:
            project_dir: Project directory
            allowed_licenses: List of allowed license types

        Returns:
            Dict with license information
        """
        try:
            cmd = ["pip-licenses", "--format=json"]

            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=project_dir, timeout=60)

            if result.returncode == 0:
                try:
                    licenses = json.loads(result.stdout) if result.stdout else []
                except json.JSONDecodeError:
                    licenses = []

                # Check for incompatible licenses
                if allowed_licenses:
                    incompatible = [
                        lic for lic in licenses
                        if lic.get('License') not in allowed_licenses
                    ]
                else:
                    incompatible = []

                return {
                    'success': True,
                    'licenses': licenses,
                    'total_packages': len(licenses),
                    'incompatible': incompatible,
                    'incompatible_count': len(incompatible)
                }
            else:
                return {"error": "License check failed"}

        except FileNotFoundError:
            return {"error": "pip-licenses not installed. Install with: pip install pip-licenses"}
        except subprocess.TimeoutExpired:
            return {"error": "License check timed out"}
