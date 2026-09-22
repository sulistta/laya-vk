"""Snake policies: shared decision guard, unified Laya agent, planner fallback."""

import math
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version

from .game import DIRECTIONS


def _versions(*names):
    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except Exception:
            pass
    return found


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


def build_state_questions(game, prompt):
    moves = game.moves()
    safe = [m for m in moves if m.safe]
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    reachable, space = game.food_reachability()
    descriptions = {}
    for move in moves:
        if not move.legal:
            descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
        elif not move.safe:
            descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
        elif move.eats:
            descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
        elif move.direction == preferred:
            descriptions[move.direction] = "Safe. Best progress toward food."
        else:
            descriptions[move.direction] = "Safe but less progress toward food."
    state = (
        f"Snake game. {len(safe)} safe directions available. "
        f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
        f"Open cells: {space}. Snake length: {len(game.body)}. "
        f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
    )
    questions = {
        "move": {
            "type": "choice",
            "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
            "criteria": descriptions,
        },
        "risk": {
            "type": "noul",
            "instructions": "Is there a safe route forward for the snake?",
        },
        "food": {
            "type": "noul",
            "instructions": "Is food reachable through the currently empty cells?",
        },
    }
    if prompt == "compact":
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        questions["move"]["instructions"] = "Choose the best safe move toward food."
        questions["move"]["criteria"] = {
            m.direction: (
                "Blocked. Collision."
                if not m.legal
                else "Unsafe. Traps the snake."
                if not m.safe
                else "Safe. Eat food now. Best."
                if m.eats
                else "Safe. Best route to food."
                if m.direction == preferred
                else "Safe. Slower route."
            )
            for m in moves
        }
        questions["risk"]["instructions"] = "Is a safe route available?"
        questions["food"]["instructions"] = "Is food reachable through empty cells?"
    return state, questions, moves, safe, preferred, reachable


def guarded_decision(game, probabilities, risk_noul, food_noul, inference_ms,
                     decision_ms, input_tokens, guarded):
    proposed = max(DIRECTIONS, key=probabilities.__getitem__)
    allowed = [m.direction for m in game.moves() if m.safe]
    executed = (
        max(allowed, key=probabilities.__getitem__)
        if guarded and proposed not in allowed and allowed
        else proposed
    )
    moves = game.moves()
    safe = [m for m in moves if m.safe]
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    return Decision(
        probabilities=probabilities,
        proposed=proposed,
        executed=executed,
        safe_directions=allowed,
        intervened=proposed != executed,
        dead_end_risk=1 - risk_noul,
        food_reachable=food_noul,
        inference_ms=inference_ms,
        decision_ms=decision_ms,
        input_tokens=input_tokens,
        output_tokens=0,
        safe_count=len(safe),
        planner_best=preferred,
    )


class PlannerPolicy:
    """Zero-dependency fallback: follow the cycle-safe planner directly."""

    def __init__(self, *, guarded=True, prompt="compact"):
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.guarded = guarded
        self.prompt = prompt
        self.metadata = {
            "name": "hamiltonian-planner (no model)",
            "hardware": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": _versions("numpy", "rich", "pillow"),
            "network": "offline",
            "policy": "Cycle-safe planner directly; no neural inference",
            "engine": "planner · CPU",
        }

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, _ = game.food_reachability()
        probabilities = {d: (1.0 if d == preferred else 0.0) for d in DIRECTIONS}
        now = time.perf_counter()
        return guarded_decision(
            game,
            probabilities,
            1.0 if safe else 0.0,
            1.0 if reachable else 0.0,
            (now - started) * 1000,
            (time.perf_counter() - started) * 1000,
            0,
            self.guarded,
        )


class LayaPolicy:
    """Laya decisions via ``laya.load`` (Vulkan auto-fallback → CPU torch)."""

    DEFAULT_MODEL = "convaiinnovations/laya"
    DEFAULT_SUBFOLDER = "multilingual"

    def __init__(self, model=None, *, subfolder=None, device=None,
                 guarded=True, prompt="compact"):
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as error:
            raise ImportError(
                "Laya needs torch + transformers + safetensors. "
                "Install CPU wheels with: "
                "uv pip install --index-url https://download.pytorch.org/whl/cpu torch"
            ) from error
        import laya

        self.model_id = model or self.DEFAULT_MODEL
        self.subfolder = self.DEFAULT_SUBFOLDER if model is None and subfolder is None else subfolder
        print(
            f"Loading {self.model_id}"
            + (f" [{self.subfolder}]" if self.subfolder else "")
            + " (Vulkan if available, else CPU); first run downloads weights...",
            flush=True,
        )
        self.agent = laya.load(self.model_id, device=device, subfolder=self.subfolder)
        self.guarded = guarded
        self.prompt = prompt
        engine = str(getattr(self.agent, "backend", self.agent.device))
        self.metadata = {
            "name": self.model_id + (f"/{self.subfolder}" if self.subfolder else ""),
            "hardware": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": _versions(
                "torch", "transformers", "numpy", "tokenizers", "huggingface-hub",
                "iree-base-runtime",
            ),
            "network": "online-first-run",
            "policy": "Laya probabilities over planner features; optional cycle safety shield",
            "engine": engine,
        }

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        state, questions, _, _, _, _ = build_state_questions(game, self.prompt)
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        return guarded_decision(
            game,
            probabilities,
            answers["risk"]["noul"],
            answers["food"]["noul"],
            inference_ms,
            (time.perf_counter() - started) * 1000,
            output["usage"]["input_tokens"],
            self.guarded,
        )
