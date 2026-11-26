"""
FastAPI Backend Gateway for NOVA (Neural Orchestration & Virtual Agent).

Exposes non-blocking REST endpoints for:
- Incident submission (`POST /incidents/trigger`)
- State & telemetry polling (`GET /incidents/{incident_id}/state`)
- Human-in-the-Loop approval/rejection (`POST /incidents/{incident_id}/approve`)
- Incident catalog listing (`GET /incidents`)
- Scenario injection for operator testing (`POST /scenarios/inject`)
- System health & checkpointer status (`GET /health`)
"""

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nova.api.schemas import (
    ApprovalSubmitRequest,
    ApprovalSubmitResponse,
    HealthResponse,
    IncidentListItem,
    IncidentStateResponse,
    IncidentTriggerRequest,
    IncidentTriggerResponse,
    ScenarioInjectRequest,
    ScenarioInjectResponse,
)
from nova.api.service import ConflictError, IncidentService
from nova.config import settings
from nova.environments.mock_workflow_runner import ScenarioType
from nova.observability import telemetry

# Initialize FastAPI application
app = FastAPI(
    title="NOVA API Gateway",
    description="Enterprise Workflow Operations & Autonomous Incident Remediation Gateway",
    version="0.8.0",
)

# CORS configuration for Streamlit operator dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance singleton
_service_instance: Optional[IncidentService] = None


def get_incident_service() -> IncidentService:
    """Dependency provider for IncidentService singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = IncidentService()
    return _service_instance


def set_incident_service(service: IncidentService) -> None:
    """Override provider for testing."""
    global _service_instance
    _service_instance = service


# ==============================================================================
# Exception Handlers
# ==============================================================================

@app.exception_handler(PermissionError)
async def permission_error_handler(request, exc: PermissionError):
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc), "error_type": "RBAC_PERMISSION_DENIED"},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "INVALID_REQUEST_PAYLOAD"},
    )


@app.exception_handler(ConflictError)
async def conflict_error_handler(request, exc: ConflictError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc), "error_type": "CONFLICT_OR_DUPLICATE"},
    )


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(service: IncidentService = Depends(get_incident_service)) -> HealthResponse:
    """System health check and runtime configuration status."""
    return HealthResponse(
        status="HEALTHY",
        version="0.8.0",
        environment=settings.nova_env,
        checkpointer=service.checkpointer.__class__.__name__,
        active_incidents_count=len(service.list_incidents()),
        telemetry_summary=telemetry.get_summary(),
    )


@app.post(
    "/incidents/trigger",
    response_model=IncidentTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Incidents"],
)
def trigger_incident(
    request: IncidentTriggerRequest,
    service: IncidentService = Depends(get_incident_service),
) -> IncidentTriggerResponse:
    """
    Submits an operational incident for autonomous investigation and remediation.
    Returns immediately with incident ID and status (non-blocking).
    """
    try:
        return service.trigger_incident(request)
    except ConflictError as ce:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ce),
        )


@app.get(
    "/incidents/{incident_id}/state",
    response_model=IncidentStateResponse,
    tags=["Incidents"],
)
def get_incident_state(
    incident_id: str,
    service: IncidentService = Depends(get_incident_service),
) -> IncidentStateResponse:
    """
    Polls the current state, autonomous investigation history, remediation plan,
    execution telemetry, and any pending Human-in-the-Loop authorization request.
    """
    state = service.get_incident_state(incident_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found.",
        )
    return state


@app.post(
    "/incidents/{incident_id}/approve",
    response_model=ApprovalSubmitResponse,
    tags=["Incidents"],
)
def submit_approval(
    incident_id: str,
    request: ApprovalSubmitRequest,
    service: IncidentService = Depends(get_incident_service),
) -> ApprovalSubmitResponse:
    """
    Submits a human operator's authorization decision for a pending destructive action.
    Validates approver role against strict RBAC rules and resumes execution asynchronously.
    """
    try:
        return service.submit_approval(incident_id, request)
    except ConflictError as ce:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(ce),
        )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(pe),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )


@app.get("/incidents", response_model=List[IncidentListItem], tags=["Incidents"])
def list_incidents(
    service: IncidentService = Depends(get_incident_service),
) -> List[IncidentListItem]:
    """Lists all active and historical incidents with status summaries."""
    return service.list_incidents()


@app.post(
    "/scenarios/inject",
    response_model=ScenarioInjectResponse,
    tags=["Testing & Demos"],
)
def inject_scenario(
    request: ScenarioInjectRequest,
    service: IncidentService = Depends(get_incident_service),
) -> ScenarioInjectResponse:
    """
    Injects a synthetic operational scenario into the simulated enterprise environment
    and executes the initial batch run to generate realistic failure telemetry.
    """
    try:
        scenario_type = ScenarioType(request.scenario.upper())
    except ValueError:
        valid_scenarios = [s.value for s in ScenarioType]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scenario '{request.scenario}'. Must be one of: {valid_scenarios}",
        )

    # Inject scenario under IncidentService lock (blocks reset during active incident / pending approval)
    state = service.inject_scenario(scenario_type)

    failed_job = None
    error_msg = None
    for j in state.jobs.values():
        if j.status.value == "FAILED":
            failed_job = j.id
            error_msg = j.error_message
            break

    return ScenarioInjectResponse(
        workflow_id=state.workflow_id,
        run_id=state.run_id,
        scenario=request.scenario,
        failed_job=failed_job,
        error_message=error_msg,
        active_locks=state.active_locks,
    )
