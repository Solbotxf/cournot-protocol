"""
PAN Runtime — minimal ENCOMPASS-inspired search over execution paths.

Implements the core idea from the PAN/ENCOMPASS paper: treat unreliable
operations (LLM calls) as branchpoints and use search algorithms to find
the best execution path through a workflow.

Primitives:
    - Checkpoint: snapshot of execution state at a branchpoint.
    - record_score: attach a scalar score to the current state.
    - SearchConfig: algorithm + hyperparameters.

Search algorithms:
    - global_best_of_n (GBoN): run workflow N times end-to-end, keep best.
    - local_best_of_n (LBoN): at each branchpoint, sample N, keep best-1.
    - beam: keep top-K partial paths across branchpoints.

Usage:
    The collector workflow is written as a generator that yields
    ``BranchRequest`` objects at each unreliable step.  The runtime
    resumes the generator with the chosen candidate for that branch.

    The workflow must NOT contain retry loops or "try N times" logic.
    It is written *as if* the LLM always succeeds.  The runtime handles
    exploration of alternatives.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Branch request (yielded by the workflow generator)
# ---------------------------------------------------------------------------

@dataclass
class BranchRequest:
    """Yielded by the workflow at each branchpoint.

    The workflow yields this to say: "I need the runtime to produce
    ``n_candidates`` alternatives for this step.  Here is the function
    that generates one candidate and the scoring function."

    Attributes:
        tag: human-readable label (e.g. "query_generation").
        generate_fn: callable that produces one candidate output.
            Signature: ``() -> Any``.  Called N times by the runtime.
        score_fn: callable that evaluates a candidate.
            Signature: ``(candidate) -> float``.
        n_candidates: how many alternatives to sample at this branch.
            The runtime may override this from SearchConfig.default_branching.
    """

    tag: str
    generate_fn: Callable[[], Any]
    score_fn: Callable[[Any], float]
    n_candidates: int | None = None  # None → use SearchConfig.default_branching


# ---------------------------------------------------------------------------
# Score record
# ---------------------------------------------------------------------------

@dataclass
class ScoreRecord:
    """A scored checkpoint in an execution path."""
    tag: str
    value: float


# ---------------------------------------------------------------------------
# Execution path (accumulated state while running the generator)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPath:
    """One (possibly partial) execution path through the workflow."""

    choices: list[Any] = field(default_factory=list)
    scores: list[ScoreRecord] = field(default_factory=list)
    result: Any = None
    finished: bool = False
    error: str | None = None

    @property
    def total_score(self) -> float:
        return sum(s.value for s in self.scores)

    @property
    def final_score(self) -> float:
        """The last recorded score — the primary optimisation objective."""
        if not self.scores:
            return 0.0
        return self.scores[-1].value

    @property
    def score_breakdown(self) -> dict[str, float]:
        return {s.tag: s.value for s in self.scores}


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

class SearchAlgo(str, Enum):
    BON_GLOBAL = "bon_global"
    BON_LOCAL = "bon_local"
    BEAM = "beam"


@dataclass
class SearchConfig:
    """Configuration for the search algorithm.

    Attributes:
        algo: which search algorithm to use.
        default_branching: default N (candidates per branchpoint).
        beam_width: K — how many partial paths to keep (beam only).
        max_expansions: safety cap on total generate_fn calls.
        seed: optional RNG seed for determinism.
    """

    algo: SearchAlgo = SearchAlgo.BON_GLOBAL
    default_branching: int = 3
    beam_width: int = 2
    max_expansions: int = 50
    seed: int | None = None


# ---------------------------------------------------------------------------
# Workflow type alias
# ---------------------------------------------------------------------------

# A workflow factory takes no arguments and returns a generator that yields
# BranchRequests and returns the final result via StopIteration.value.
WorkflowFactory = Callable[[], Generator[BranchRequest, Any, T]]


# ---------------------------------------------------------------------------
# Search runner
# ---------------------------------------------------------------------------

def search(
    workflow_factory: WorkflowFactory[T],
    config: SearchConfig,
) -> tuple[T, ExecutionPath]:
    """Run the search algorithm over a workflow and return (best_result, best_path).

    The workflow is a generator factory — we call it to get a fresh generator
    for each execution path we need to explore.

    Returns:
        (result, path) where result is the workflow's return value on the
        best execution path found, and path contains scores + choices.
    """
    rng = random.Random(config.seed)

    if config.algo == SearchAlgo.BON_GLOBAL:
        return _search_bon_global(workflow_factory, config, rng)
    elif config.algo == SearchAlgo.BON_LOCAL:
        return _search_bon_local(workflow_factory, config, rng)
    elif config.algo == SearchAlgo.BEAM:
        return _search_beam(workflow_factory, config, rng)
    else:
        raise ValueError(f"Unknown search algorithm: {config.algo}")


# ---------------------------------------------------------------------------
# Global Best-of-N
# ---------------------------------------------------------------------------

def _search_bon_global(
    wf: WorkflowFactory[T],
    config: SearchConfig,
    rng: random.Random,
) -> tuple[T, ExecutionPath]:
    """Sample N full runs end-to-end, return the one with highest final_score."""
    n = config.default_branching
    best_path: ExecutionPath | None = None
    best_result: Any = None
    expansions = 0

    for _ in range(n):
        if expansions >= config.max_expansions:
            break
        path = ExecutionPath()
        gen = wf()
        try:
            branch = next(gen)
            while True:
                # At each branchpoint, sample ONE candidate (no branching)
                if expansions >= config.max_expansions:
                    break
                candidate = branch.generate_fn()
                expansions += 1
                score = branch.score_fn(candidate)
                path.choices.append(candidate)
                path.scores.append(ScoreRecord(tag=branch.tag, value=score))
                branch = gen.send(candidate)
        except StopIteration as e:
            path.result = e.value
            path.finished = True
        except Exception as e:
            path.error = str(e)

        if path.finished and (best_path is None or path.final_score > best_path.final_score):
            best_path = path
            best_result = path.result

    if best_path is None:
        # All paths failed — run once more without safety cap
        path = ExecutionPath()
        gen = wf()
        try:
            branch = next(gen)
            while True:
                candidate = branch.generate_fn()
                score = branch.score_fn(candidate)
                path.choices.append(candidate)
                path.scores.append(ScoreRecord(tag=branch.tag, value=score))
                branch = gen.send(candidate)
        except StopIteration as e:
            path.result = e.value
            path.finished = True
        except Exception as e:
            path.error = str(e)
        best_path = path
        best_result = path.result

    return best_result, best_path


# ---------------------------------------------------------------------------
# Local Best-of-N (beam width=1, branching=N)
# ---------------------------------------------------------------------------

def _search_bon_local(
    wf: WorkflowFactory[T],
    config: SearchConfig,
    rng: random.Random,
) -> tuple[T, ExecutionPath]:
    """At each branchpoint, sample N candidates and keep the best one."""
    n = config.default_branching
    path = ExecutionPath()
    gen = wf()
    expansions = 0

    try:
        branch = next(gen)
        while True:
            branch_n = branch.n_candidates or n
            best_candidate = None
            best_score = float("-inf")

            for _ in range(branch_n):
                if expansions >= config.max_expansions:
                    break
                candidate = branch.generate_fn()
                expansions += 1
                score = branch.score_fn(candidate)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            path.choices.append(best_candidate)
            path.scores.append(ScoreRecord(tag=branch.tag, value=best_score))
            branch = gen.send(best_candidate)
    except StopIteration as e:
        path.result = e.value
        path.finished = True
    except Exception as e:
        path.error = str(e)

    return path.result, path


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@dataclass
class _BeamState:
    """Internal state for one beam in beam search."""
    choices: list[Any] = field(default_factory=list)
    scores: list[ScoreRecord] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return sum(s.value for s in self.scores)


def _search_beam(
    wf: WorkflowFactory[T],
    config: SearchConfig,
    rng: random.Random,
) -> tuple[T, ExecutionPath]:
    """Beam search: keep top-K partial paths across branchpoints."""
    n = config.default_branching
    k = config.beam_width
    expansions = 0

    # Discover branchpoints by running one path to find the structure.
    # We'll use a replay approach: collect branchpoints, then explore.
    #
    # Strategy: run the workflow step by step.  At each branchpoint,
    # expand all current beams × N candidates, keep top-K by cumulative score.

    # Start with one empty beam
    beams: list[_BeamState] = [_BeamState()]
    branch_index = 0

    # We run a "probe" generator to discover branchpoints one at a time,
    # then replay for each beam.
    #
    # Simpler approach: maintain generators for all active beams.
    # At each step, advance all generators, expand, prune.

    # Initialize generators for each beam
    generators: list[Generator] = [wf()]
    pending_branches: list[BranchRequest | None] = [None]

    # Advance each generator to first branchpoint
    for i, gen in enumerate(generators):
        try:
            pending_branches[i] = next(gen)
        except StopIteration as e:
            # Workflow finished immediately with no branchpoints
            path = ExecutionPath(
                choices=beams[i].choices,
                scores=beams[i].scores,
                result=e.value,
                finished=True,
            )
            return e.value, path

    while any(b is not None for b in pending_branches):
        if expansions >= config.max_expansions:
            break

        new_beams: list[_BeamState] = []
        new_generators: list[Generator] = []
        new_pending: list[BranchRequest | None] = []

        for i, branch in enumerate(pending_branches):
            if branch is None:
                # This beam already finished; carry it forward
                new_beams.append(beams[i])
                new_generators.append(generators[i])
                new_pending.append(None)
                continue

            branch_n = branch.n_candidates or n

            # Generate N candidates for this beam
            candidates: list[tuple[Any, float]] = []
            for _ in range(branch_n):
                if expansions >= config.max_expansions:
                    break
                candidate = branch.generate_fn()
                expansions += 1
                score = branch.score_fn(candidate)
                candidates.append((candidate, score))

            # Create expanded beams
            for candidate, score in candidates:
                new_state = _BeamState(
                    choices=beams[i].choices + [candidate],
                    scores=beams[i].scores + [ScoreRecord(tag=branch.tag, value=score)],
                )
                new_beams.append(new_state)
                # We need a fresh generator replayed up to this point
                new_generators.append(None)  # placeholder
                new_pending.append(branch)  # will be advanced below

        # Prune to top-K by total_score
        if len(new_beams) > k:
            indexed = list(enumerate(new_beams))
            indexed.sort(key=lambda x: x[1].total_score, reverse=True)
            keep_indices = set(idx for idx, _ in indexed[:k])
            pruned_beams = []
            pruned_pending = []
            for idx, beam in indexed[:k]:
                pruned_beams.append(beam)
                pruned_pending.append(new_pending[idx])
            new_beams = pruned_beams
            new_pending = pruned_pending

        # Now replay generators for kept beams and advance to next branchpoint
        final_generators: list[Generator] = []
        final_pending: list[BranchRequest | None] = []

        for i, beam in enumerate(new_beams):
            if new_pending[i] is None:
                # Already finished
                final_generators.append(None)  # type: ignore
                final_pending.append(None)
                continue

            # Replay generator with this beam's choices
            gen = wf()
            try:
                br = next(gen)
                for choice in beam.choices[:-1]:
                    br = gen.send(choice)
                # Send the last choice and get next branchpoint
                br = gen.send(beam.choices[-1])
                final_generators.append(gen)
                final_pending.append(br)
            except StopIteration as e:
                # Workflow finished after sending last choice
                beam_as_path = ExecutionPath(
                    choices=beam.choices,
                    scores=beam.scores,
                    result=e.value,
                    finished=True,
                )
                final_generators.append(None)  # type: ignore
                final_pending.append(None)
                # Store result on beam for final selection
                new_beams[i] = beam  # keep as-is

        beams = new_beams
        generators = final_generators
        pending_branches = final_pending

    # Select best finished beam
    best_idx = 0
    best_score = float("-inf")
    for i, beam in enumerate(beams):
        if beam.total_score > best_score:
            best_score = beam.total_score
            best_idx = i

    # Replay the best beam to get the final result
    best_beam = beams[best_idx]
    gen = wf()
    result = None
    try:
        br = next(gen)
        for choice in best_beam.choices:
            br = gen.send(choice)
    except StopIteration as e:
        result = e.value

    path = ExecutionPath(
        choices=best_beam.choices,
        scores=best_beam.scores,
        result=result,
        finished=True,
    )
    return result, path
