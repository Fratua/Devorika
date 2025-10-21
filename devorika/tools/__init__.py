"""
Tools and utilities for Devorika.

This package contains all the tools that Devorika can use to perform various tasks.
Tools are organized by category for better maintainability.
"""

from .base import Tool, ToolRegistry

# File operations
from .file_tools import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirectoryTool,
    SearchCodeTool
)

# Code execution
from .execution_tools import (
    BashTool,
    PythonExecuteTool,
    InstallPackageTool,
    RunTestsTool
)

# Code analysis
from .code_analysis import (
    AnalyzeCodeTool,
    FindBugsTool,
    GetCodeComplexityTool
)

# Git operations
from .git_tools import (
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitLogTool,
    GitBranchTool,
    GitPushTool
)

# Web operations
from .web_tools import (
    WebSearchTool,
    FetchURLTool,
    ReadDocumentationTool
)

# IDE integration tools
from .ide_tools import (
    CodeIntelligenceTool,
    RefactoringTool,
    CodeFormatterTool,
    LiveLintTool,
    SymbolNavigationTool,
    CodeCompletionTool
)

# Database tools
from .database_tools import (
    SQLQueryTool,
    DatabaseSchemaTool,
    MigrationTool,
    ORMModelTool,
    NoSQLTool
)

# DevOps and cloud tools
from .devops_tools import (
    DockerTool,
    KubernetesTool,
    TerraformTool,
    CICDTool
)

# Security tools
from .security_tools import (
    VulnerabilityScannerTool,
    SecretDetectionTool,
    SASTTool,
    ComplianceCheckerTool,
    LicenseCheckerTool
)

# Performance tools
from .performance_tools import (
    CPUProfilerTool,
    MemoryProfilerTool,
    PerformanceBenchmarkTool,
    LoadTesterTool,
    CodeOptimizationTool
)

# API development tools
from .api_tools import (
    APIScaffoldTool,
    OpenAPIGeneratorTool,
    APITestGeneratorTool,
    APIClientGeneratorTool
)

# ML/AI tools
from .ml_tools import (
    MLModelTrainerTool,
    HyperparameterTunerTool,
    ModelEvaluatorTool,
    FeatureEngineeringTool,
    ModelDeploymentTool,
    DataAnalysisTool
)

# Advanced analysis tools
from .advanced_analysis_tools import (
    DependencyGraphTool,
    ImpactAnalysisTool,
    ArchitectureAnalyzerTool,
    CodeQualityMetricsTool,
    CodeDuplicationDetectorTool
)

# Export all tools
__all__ = [
    # Base
    'Tool',
    'ToolRegistry',
    # File operations
    'ReadFileTool',
    'WriteFileTool',
    'EditFileTool',
    'ListDirectoryTool',
    'SearchCodeTool',
    # Code execution
    'BashTool',
    'PythonExecuteTool',
    'InstallPackageTool',
    'RunTestsTool',
    # Code analysis
    'AnalyzeCodeTool',
    'FindBugsTool',
    'GetCodeComplexityTool',
    # Git operations
    'GitStatusTool',
    'GitDiffTool',
    'GitCommitTool',
    'GitLogTool',
    'GitBranchTool',
    'GitPushTool',
    # Web operations
    'WebSearchTool',
    'FetchURLTool',
    'ReadDocumentationTool',
    # IDE integration
    'CodeIntelligenceTool',
    'RefactoringTool',
    'CodeFormatterTool',
    'LiveLintTool',
    'SymbolNavigationTool',
    'CodeCompletionTool',
    # Database
    'SQLQueryTool',
    'DatabaseSchemaTool',
    'MigrationTool',
    'ORMModelTool',
    'NoSQLTool',
    # DevOps and cloud
    'DockerTool',
    'KubernetesTool',
    'TerraformTool',
    'CICDTool',
    # Security
    'VulnerabilityScannerTool',
    'SecretDetectionTool',
    'SASTTool',
    'ComplianceCheckerTool',
    'LicenseCheckerTool',
    # Performance
    'CPUProfilerTool',
    'MemoryProfilerTool',
    'PerformanceBenchmarkTool',
    'LoadTesterTool',
    'CodeOptimizationTool',
    # API development
    'APIScaffoldTool',
    'OpenAPIGeneratorTool',
    'APITestGeneratorTool',
    'APIClientGeneratorTool',
    # ML/AI
    'MLModelTrainerTool',
    'HyperparameterTunerTool',
    'ModelEvaluatorTool',
    'FeatureEngineeringTool',
    'ModelDeploymentTool',
    'DataAnalysisTool',
    # Advanced analysis
    'DependencyGraphTool',
    'ImpactAnalysisTool',
    'ArchitectureAnalyzerTool',
    'CodeQualityMetricsTool',
    'CodeDuplicationDetectorTool',
]


def register_all_tools():
    """
    Register all available tools with the ToolRegistry.
    This function is called during initialization to make all tools available.
    """
    registry = ToolRegistry()

    # File operations (5 tools)
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchCodeTool())

    # Code execution (4 tools)
    registry.register(BashTool())
    registry.register(PythonExecuteTool())
    registry.register(InstallPackageTool())
    registry.register(RunTestsTool())

    # Code analysis (3 tools)
    registry.register(AnalyzeCodeTool())
    registry.register(FindBugsTool())
    registry.register(GetCodeComplexityTool())

    # Git operations (6 tools)
    registry.register(GitStatusTool())
    registry.register(GitDiffTool())
    registry.register(GitCommitTool())
    registry.register(GitLogTool())
    registry.register(GitBranchTool())
    registry.register(GitPushTool())

    # Web operations (3 tools)
    registry.register(WebSearchTool())
    registry.register(FetchURLTool())
    registry.register(ReadDocumentationTool())

    # IDE integration (6 tools)
    registry.register(CodeIntelligenceTool())
    registry.register(RefactoringTool())
    registry.register(CodeFormatterTool())
    registry.register(LiveLintTool())
    registry.register(SymbolNavigationTool())
    registry.register(CodeCompletionTool())

    # Database (5 tools)
    registry.register(SQLQueryTool())
    registry.register(DatabaseSchemaTool())
    registry.register(MigrationTool())
    registry.register(ORMModelTool())
    registry.register(NoSQLTool())

    # DevOps and cloud (4 tools)
    registry.register(DockerTool())
    registry.register(KubernetesTool())
    registry.register(TerraformTool())
    registry.register(CICDTool())

    # Security (5 tools)
    registry.register(VulnerabilityScannerTool())
    registry.register(SecretDetectionTool())
    registry.register(SASTTool())
    registry.register(ComplianceCheckerTool())
    registry.register(LicenseCheckerTool())

    # Performance (5 tools)
    registry.register(CPUProfilerTool())
    registry.register(MemoryProfilerTool())
    registry.register(PerformanceBenchmarkTool())
    registry.register(LoadTesterTool())
    registry.register(CodeOptimizationTool())

    # API development (4 tools)
    registry.register(APIScaffoldTool())
    registry.register(OpenAPIGeneratorTool())
    registry.register(APITestGeneratorTool())
    registry.register(APIClientGeneratorTool())

    # ML/AI (6 tools)
    registry.register(MLModelTrainerTool())
    registry.register(HyperparameterTunerTool())
    registry.register(ModelEvaluatorTool())
    registry.register(FeatureEngineeringTool())
    registry.register(ModelDeploymentTool())
    registry.register(DataAnalysisTool())

    # Advanced analysis (5 tools)
    registry.register(DependencyGraphTool())
    registry.register(ImpactAnalysisTool())
    registry.register(ArchitectureAnalyzerTool())
    registry.register(CodeQualityMetricsTool())
    registry.register(CodeDuplicationDetectorTool())

    return registry


# Tool count summary
TOOL_CATEGORIES = {
    'File Operations': 5,
    'Code Execution': 4,
    'Code Analysis': 3,
    'Git Operations': 6,
    'Web Operations': 3,
    'IDE Integration': 6,
    'Database': 5,
    'DevOps & Cloud': 4,
    'Security': 5,
    'Performance': 5,
    'API Development': 4,
    'ML/AI': 6,
    'Advanced Analysis': 5,
}

TOTAL_TOOLS = sum(TOOL_CATEGORIES.values())  # 61 tools total!
