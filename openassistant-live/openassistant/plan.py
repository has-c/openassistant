"""Bound the plan tool's output before sending it to the browser."""


def plan_steps(value):
    if not isinstance(value, list) or len(value) > 20:
        raise ValueError("A plan must contain at most 20 steps")
    result = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("step"), str)
            or not 0 < len(item["step"].strip()) <= 240
            or item.get("status") not in {"pending", "in_progress", "completed"}
        ):
            raise ValueError("Invalid plan step")
        result.append({"step": item["step"], "status": item["status"]})
    if sum(item["status"] == "in_progress" for item in result) > 1:
        raise ValueError("Only one step can be in progress")
    return result
