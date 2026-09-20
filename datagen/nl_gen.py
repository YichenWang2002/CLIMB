"""Generate natural-language `input` for each sampled task via DeepSeek.

Hard contract: the output is PROSE ONLY. No PDDL/STRIPS, no XML, no logic
symbols. Entity ids appear naturally in parentheses on first mention.
A regex leak filter rejects (and regenerates) any violation.
"""
from __future__ import annotations

import re

from common.llm import chat_batch
from .names import LOC_EN, ITEM_EN, SCENE_EN, UNIQUE_ACTION_EN, loc_en, item_en, robot_en

# formalism leak patterns -> reject
LEAK_PATTERNS = [
    r"\(:", r"\bPDDL\b", r"\bSTRIPS\b", r"\bprecondition", r"\bpredicate",
    r"∀", r"∃", r"∧", r"∨", r"¬", r"→", r"<root", "<Sequence", "<Fallback",
    r"<BehaviorTree", r"\bitem_at\b", r"\bcan_reach\b", r"\bco_carrying\b",
    r"\bcarrying\(", r"\bat\(", r"\bconnected\(",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)

STYLES = [
    "a direct, imperative mission briefing",
    "a narrative scenario description told like a short story setup",
    "a casual request, as if a facility manager is talking to the robot team",
]

INSTRUCTION = (
    "You are a helpful assistant that creates behavior trees for multi-robot teams.\n"
    "Your task:\n"
    "- Convert the provided natural-language mission description into an XML-formatted "
    "multi-agent behavior tree.\n"
    "- The behavior tree must be compatible with the BehaviorTree.CPP library "
    "(<root BTCPP_format=\"4\">), with one <BehaviorTree> per robot coordinated by a "
    "main tree.\n"
    "- Use Fallback nodes to recover from the failures mentioned in the mission.\n"
    "- Use SignalReady/WaitReady and the Handover/Co* joint nodes for inter-robot "
    "coordination.\n\n"
    "Output Requirements:\n"
    "- Output only the XML representation of the behavior tree. Do not include "
    "explanations, comments, or any additional text.\n"
    "- Every robot mentioned in the mission must have its own <BehaviorTree ID=\"Agent_<name>\">."
)


def _facts_summary(task: dict) -> str:
    """Structured-but-plain English summary of the task for the LLM prompt."""
    dom = task["domain"]
    init, goal = task["init_dynamic"], task["goal"]
    lines = []
    robots = task["robots"]
    lines.append("Robots: " + ", ".join(f"{robot_en(r)} (id {r})" for r in robots))
    # placement
    for f in init:
        if f[0] == "at":
            lines.append(f"- {robot_en(f[1])} starts at {loc_en(dom, f[2])} ({f[2]}).")
        elif f[0] == "item_at":
            heavy = ("heavy", f[1]) in init
            note = " [heavy: needs two robots to carry]" if heavy else ""
            lines.append(f"- {item_en(dom, f[1])} ({f[1]}) is at {loc_en(dom, f[2])} ({f[2]}).{note}")
    # goals
    gl = []
    for f in goal:
        if f[0] == "item_at":
            gl.append(f"bring {item_en(dom, f[1])} ({f[1]}) to {loc_en(dom, f[2])} ({f[2]})")
        elif f[0] == "cataloged":
            gl.append(f"catalog {item_en(dom, f[1])} ({f[1]}) using CatalogBook")
        elif f[0] == "watered":
            gl.append(f"water {loc_en(dom, f[1])} ({f[1]}) using WaterPlants")
        elif f[0] == "inspected":
            gl.append(f"inspect {loc_en(dom, f[1])} ({f[1]}) using InspectPlants")
        elif f[0] == "sanitized":
            gl.append(f"sanitize the counter at {loc_en(dom, f[1])} ({f[1]}) using SanitizeCounter")
    lines.append("Mission goals: " + "; ".join(gl) + ".")
    # zones / reachability
    zparts = []
    for r, zone in task["zones"].items():
        if len(zone) <= 3:
            zparts.append(f"{robot_en(r)} can only operate at "
                          + " and ".join(f"{loc_en(dom, l)} ({l})" for l in zone))
    if zparts:
        lines.append("Operating areas: " + "; ".join(zparts) + ".")
    # faults
    for f in task["faults"]:
        if f["type"] == "blocked_edge":
            a, b = f["edge"]
            lines.append(f"Hazard: the passage between {loc_en(dom, a)} ({a}) and "
                         f"{loc_en(dom, b)} ({b}) may be blocked by debris; a robot at "
                         f"{loc_en(dom, a)} ({a}) can clear it with ClearPath before passing.")
        elif f["type"] == "battery":
            lines.append(f"Hazard: {robot_en(f['robot'])} has a weak battery that dies after "
                         f"{f['budget']} moves; it must then reach a charging station and "
                         f"Recharge before continuing.")
        elif f["type"] == "handover_fail":
            lines.append(f"Hazard: the first handover attempt of {item_en(dom, f['item'])} "
                         f"({f['item']}) between {robot_en(f['giver'])} and "
                         f"{robot_en(f['receiver'])} may fumble; be ready to retry it.")
    # unique actions for test domains
    used = sorted({u for u, pred in (("CatalogBook", "cataloged"), ("WaterPlants", "watered"),
                                     ("InspectPlants", "inspected"),
                                     ("SanitizeCounter", "sanitized"))
                   if any(g[0] == pred for g in goal)})
    for u in used:
        lines.append(f"Extra skill: the robots have {UNIQUE_ACTION_EN[u]}.")
    # adjacency (plain English)
    adj = {}
    for a, b in task["connected"]:
        adj.setdefault(a, set()).add(b)
    conn = []
    for a in sorted(adj):
        conn.append(f"{loc_en(dom, a)} ({a}) connects to "
                    + ", ".join(f"{loc_en(dom, b)} ({b})" for b in sorted(adj[a])))
    lines.append("Layout: " + "; ".join(conn) + ".")
    stations = task["charge_stations"]
    lines.append("Charging stations: " + ", ".join(f"{loc_en(dom, s)} ({s})" for s in stations) + ".")
    return "\n".join(lines)


def build_prompt(task: dict, style: str) -> list:
    scene = SCENE_EN[task["domain"]]
    summary = _facts_summary(task)
    user = (
        f"Write {style} for a multi-robot mission set in {scene}.\n\n"
        "Facts you must faithfully cover (rewrite them as fluent prose, keep every id "
        "in parentheses on first mention, do not omit goals or hazards):\n"
        f"{summary}\n\n"
        "Rules:\n"
        "- 120-220 words of plain English prose only.\n"
        "- NO formal logic, NO PDDL/STRIPS notation, NO XML, NO bullet-point fact dumps "
        "with parentheses like at(robot, place).\n"
        "- Mention each location/item id once in parentheses, e.g. 'shelf A (shelf_a)'.\n"
        "- Make it sound like a real mission given to a robot team."
    )
    return [{"role": "user", "content": user}]


def generate_nl_inputs(tasks: list, max_workers: int = 8, max_rounds: int = 2) -> list:
    """Returns list of NL strings (None where generation failed after retries)."""
    styles = [STYLES[i % len(STYLES)] for i in range(len(tasks))]
    results = [None] * len(tasks)
    pending = list(range(len(tasks)))
    for round_ in range(max_rounds):
        jobs = [build_prompt(tasks[i], styles[i]) for i in pending]
        outs = chat_batch(jobs, max_workers=max_workers, temperature=0.9 if round_ == 0 else 0.4)
        still = []
        for i, text in zip(pending, outs):
            text = text.strip()
            if text and not LEAK_RE.search(text) and 60 < len(text) < 2000:
                results[i] = text
            else:
                still.append(i)
        pending = still
        if not pending:
            break
    return results
