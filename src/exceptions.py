# Shared custom exceptions for the project.
# These keep error handling explicit and make the service layer easier to test.

class CourseRAGError(Exception):
    """Base exception for all project-specific errors."""


class ArtifactNotFoundError(CourseRAGError):
    """Raised when a required artifact file is missing."""


class ArtifactValidationError(CourseRAGError):
    """Raised when artifact contents are invalid or inconsistent."""


class RetrievalError(CourseRAGError):
    """Raised when retrieval fails."""


class QueryFormulationError(CourseRAGError):
    """Raised when query formulation fails."""


class FormattingError(CourseRAGError):
    """Raised when citation or path formatting fails."""


class AgentError(CourseRAGError):
    """Raised when the agent layer fails."""


class EvaluationError(CourseRAGError):
    """Raised when evaluation logic fails."""