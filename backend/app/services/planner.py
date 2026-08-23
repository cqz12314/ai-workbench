import json
from dataclasses import dataclass
from datetime import UTC, datetime

from app.db.session import SessionLocal
from app.models import DevelopmentTask

TASK_STATUSES = {"pending", "running", "completed", "failed"}


class PlannerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskStep:
    name: str
    action: str
    status: str = "pending"


def _steps_for(task: str) -> list[TaskStep]:
    lowered = task.lower()
    steps = [TaskStep("analyze", "analyze_repository")]
    if any(word in lowered for word in ("修改", "实现", "增加", "fix", "change", "implement")):
        steps.append(TaskStep("modify", "propose_change"))
    steps.append(TaskStep("test", "run_tests"))
    return steps


def create_plan(task: str) -> dict:
    if not isinstance(task, str) or not task.strip() or len(task.strip()) > 10_000:
        raise PlannerError("Task description is invalid")
    steps = _steps_for(task.strip())
    with SessionLocal() as session:
        record = DevelopmentTask(
            task=task.strip(),
            status="pending",
            current_step=None,
            plan_json=json.dumps([step.__dict__ for step in steps], ensure_ascii=False),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        task_id = record.id
    return {
        "task_id": task_id,
        "task": task.strip(),
        "status": "pending",
        "steps": [step.__dict__ for step in steps],
    }


def get_plan(task_id: int) -> tuple[DevelopmentTask, list[TaskStep]]:
    with SessionLocal() as session:
        record = session.get(DevelopmentTask, task_id)
        if record is None:
            raise PlannerError("Task plan does not exist")
        raw_steps = json.loads(record.plan_json)
        steps = [TaskStep(**step) for step in raw_steps]
        session.expunge(record)
    return record, steps


def update_task(task_id: int, status: str, current_step: str | None = None) -> None:
    if status not in TASK_STATUSES:
        raise PlannerError("Task status is invalid")
    with SessionLocal() as session:
        record = session.get(DevelopmentTask, task_id)
        if record is None:
            raise PlannerError("Task plan does not exist")
        record.status = status
        record.current_step = current_step
        record.updated_at = datetime.now(UTC)
        session.commit()
