from collections.abc import Awaitable, Callable
from typing import Any

from app.db.session import SessionLocal
from app.models import DevelopmentRun
from app.services.planner import PlannerError, get_plan, update_task
from app.services.test_runner import TestRunnerError, run_test

MAX_REPAIR_ROUNDS = 3
SAFE_ACTIONS = {"analyze_repository", "propose_change", "apply_change", "run_tests"}


class TaskExecutorError(RuntimeError):
    pass


def _record_run(
    task_id: int,
    step: str,
    action: str,
    status: str,
    result_summary: str | None = None,
    error_summary: str | None = None,
) -> None:
    with SessionLocal() as session:
        session.add(
            DevelopmentRun(
                task_id=task_id,
                step=step,
                action=action,
                status=status,
                result_summary=result_summary,
                error_summary=error_summary,
            )
        )
        session.commit()


async def execute_task(task_id: int, test_name: str = "pytest") -> dict[str, Any]:
    try:
        record, steps = get_plan(task_id)
    except PlannerError as exc:
        raise TaskExecutorError(str(exc)) from exc
    if record.status not in {"pending", "running"}:
        raise TaskExecutorError("Task plan is not executable")
    if any(step.action not in SAFE_ACTIONS for step in steps):
        raise TaskExecutorError("Task plan contains an unsafe action")

    update_task(task_id, "running")
    results: list[dict[str, Any]] = []
    try:
        for step in steps:
            update_task(task_id, "running", step.name)
            if step.action == "run_tests":
                result = run_test(test_name)
                results.append(result)
                status = "completed" if result["passed"] else "failed"
                _record_run(task_id, step.name, step.action, status, str(result))
                if not result["passed"]:
                    update_task(task_id, "failed", step.name)
                    return {"task_id": task_id, "status": "failed", "runs": results}
            else:
                summary = "Step requires the Agent's scoped tool and confirmation"
                results.append({"step": step.name, "action": step.action, "status": "pending"})
                _record_run(task_id, step.name, step.action, "completed", summary)
        update_task(task_id, "completed", None)
    except (TestRunnerError, PlannerError) as exc:
        _record_run(
            task_id,
            steps[0].name if steps else "unknown",
            "executor",
            "failed",
            error_summary=str(exc),
        )
        update_task(task_id, "failed", None)
        raise TaskExecutorError(str(exc)) from exc
    return {"task_id": task_id, "status": "completed", "runs": results}


async def run_auto_fix_loop(
    run_tests: Callable[[], Awaitable[dict[str, Any]]],
    analyze_error: Callable[[dict[str, Any]], Awaitable[Any]],
    propose_change: Callable[[Any], Awaitable[Any]],
    apply_change: Callable[[Any], Awaitable[Any]],
    confirmed: Callable[[Any], bool],
    max_rounds: int = MAX_REPAIR_ROUNDS,
) -> dict[str, Any]:
    if max_rounds < 1 or max_rounds > MAX_REPAIR_ROUNDS:
        raise TaskExecutorError("Repair rounds are limited to three")
    history: list[dict[str, Any]] = []
    for round_number in range(1, max_rounds + 1):
        result = await run_tests()
        history.append({"round": round_number, "test": result})
        if result.get("passed") is True:
            return {"status": "completed", "rounds": history}
        diagnosis = await analyze_error(result)
        proposal = await propose_change(diagnosis)
        if not confirmed(proposal):
            return {"status": "awaiting_confirmation", "rounds": history, "proposal": proposal}
        await apply_change(proposal)
    return {"status": "failed", "rounds": history, "error": "Maximum repair rounds reached"}
