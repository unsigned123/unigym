"""contacts candidate-discovery fail-fast layer .

When a declared contact group's sensor/filter bodies fail to resolve
against the actual asset, the backends REFUSE with a
:class:`ContactCandidateError` that carries a dynamically discovered
inventory of the asset's collision-capable addressable bodies (mujoco:
MJ bodies with geoms and their collision status; isaaclab: rigid-body
prims under the robot subtree with their collision-mesh coverage) plus a
keyword-marked subset of likely candidates (tokens from the group name
and the declared patterns).  The full inventory is always included --
the likely marks are presentation, never a filter.

Error body contract (contacts_contract.md §3.1): ``error`` (the message,
full inventory inline) + attributes ``discovered`` (list of
``{prim, collision, likely}`` dicts) + ``hint`` (a ready-to-paste
``declare sensor_bodies: [...]`` suggestion built from the top likely
candidate).
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence


class ContactCandidateError(ValueError):
    """A contact body declaration failed to resolve -- with candidates.

    Attributes:
        discovered: the discovered inventory, ``[{prim, collision, likely}]``
            in asset order; ``collision`` is the backend's collision
            capability signal (mujoco: any geom with ``contype != 0``;
            isaaclab: descendant collision-mesh coverage), ``likely`` marks
            keyword matches (never a filter).
        hint: one-line fix suggestion naming the top likely candidate.
        group / arg: the failing contact group name and yaml field.
    """

    def __init__(self, message: str, discovered: list[dict], hint: str,
                 group: str, arg: str):
        super().__init__(message)
        self.discovered = discovered
        self.hint = hint
        self.group = group
        self.arg = arg


def _name_tokens(*sources) -> set[str]:
    """Lowercase keyword tokens (>= 3 chars) from names/patterns.

    Feeds the likely-candidate marks: the group name and the declared
    (unresolved) patterns carry the intent -- ``hand_crate_left`` wanting
    fingers should mark ``left_hand_index_0`` as likely.
    """
    toks: set[str] = set()
    for src in sources:
        if src is None:
            continue
        parts = [src] if isinstance(src, str) else src
        for part in parts:
            for tok in re.split(r"[^A-Za-z0-9]+", str(part).lower()):
                if len(tok) >= 3:
                    toks.add(tok)
    return toks


def _mark_likely(inventory: list[dict], tokens: set[str]) -> None:
    """Set the ``likely`` flag in place (substring match, never a filter)."""
    for entry in inventory:
        name = str(entry["prim"]).lower()
        entry["likely"] = any(t in name for t in tokens) if tokens else False


def _hint_declaration(group: str, arg: str, inventory: list[dict]) -> str:
    """A ready-to-paste declaration suggestion (best likely candidate).

    "Best" = most group/pattern tokens matched (so ``hand`` beats
    ``left``-only matches for a ``hand_crate`` group), first in asset
    order on ties; any likely candidate beats a bare likely-less list.
    """
    tokens = _name_tokens(group, arg)
    if not any(e.get("likely") for e in inventory):
        tokens = set()

    def score(entry: dict) -> tuple:
        name = str(entry["prim"]).lower()
        return (sum(1 for t in tokens if t in name),
                1 if entry.get("likely") else 0,
                -len(name))          # ties -> the more specific name

    ranked = sorted(range(len(inventory)), key=lambda i: score(inventory[i]),
                    reverse=True)
    top = inventory[ranked[0]]["prim"] if inventory else "<pick>"
    return f"declare {arg}: ['{top}']  (contact group {group!r})"


def contact_candidate_error(group: str, arg: str, patterns: Sequence[str],
                            inventory: list[dict], backend: str,
                            reason: str = "matched no bodies") \
        -> ContactCandidateError:
    """Build the enriched failure (message embeds the full inventory).

    ``inventory`` entries are ``{prim, collision}``; the likely marks are
    added here from the group/pattern tokens.  The message lists every
    candidate (likely ones starred) so the failure is actionable from the
    run log alone -- no asset spelunking required.
    """
    tokens = _name_tokens(group, arg, patterns)
    _mark_likely(inventory, tokens)
    lines = [f"contact group {group!r}: {arg} {list(patterns)} {reason} "
             f"-- refusing to guess ({backend} asset).", "discovered:"]
    for entry in inventory:
        mark = "*" if entry.get("likely") else " "
        lines.append(f"  {mark} {entry['prim']}  (collision={entry['collision']})")
    lines.append("(*) likely by name; full list above is the asset inventory")
    hint = _hint_declaration(group, arg, inventory)
    lines.append(f"hint: {hint}")
    return ContactCandidateError("\n".join(lines), inventory, hint,
                                 group, arg)


# --------------------------------------------------------------------------- #
# backend inventories
# --------------------------------------------------------------------------- #
def mujoco_body_inventory(m) -> list[dict]:
    """Every MJ body with geoms + its collision status (model order).

    ``collision`` = any geom of the body has ``contype != 0`` (visual-only
    bodies show ``collision=false``).  Bodies without geoms cannot appear
    in ``data.ncon`` and are omitted (they are never valid contact
    addresses).
    """
    out: list[dict] = []
    for i in range(1, m.nbody):                    # skip world
        name = m.body(i).name
        if not name or m.body_geomnum[i] == 0:
            continue
        adr, n = int(m.body_geomadr[i]), int(m.body_geomnum[i])
        collides = any(m.geom_contype[g] != 0
                       for g in range(adr, adr + n) if g >= 0)
        out.append({"prim": name, "collision": bool(collides)})
    return out


def usd_robot_inventory(robot) -> list[dict]:
    """Rigid-body prims under the robot subtree + collision-mesh coverage.

    The isaaclab-side discovery: walks ``robot.prim``'s subtree and lists
    every prim carrying ``UsdPhysics.RigidBodyAPI`` (the addressable
    links, nested scopes included) with ``collision`` = whether any
    descendant Mesh applies ``UsdPhysics.CollisionAPI``.  Foreign assets
    frequently leave the collision API unapplied (the dex3 gap) -- the
    inventory shows that honestly instead of hiding it.

    Falls back to the resolved ``robot.data.body_names`` (collision=true:
    articulation links are collision links by construction) when the USD
    stage is not reachable -- this is the ERROR path; discovery must not
    mask the original resolution failure with a crash of its own.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        prim = robot.prim
        root = str(prim.GetPath())
        prefix = root + "/"
        collision_paths: list[str] = []
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
                collision_paths.append(str(p.GetPath()))
        out: list[dict] = []
        for p in Usd.PrimRange(prim):
            if not p.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            path = str(p.GetPath())
            rel = path[len(root) + 1:]
            collides = any(cp.startswith(prefix + rel + "/")
                           or cp == path for cp in collision_paths)
            out.append({"prim": rel, "collision": bool(collides)})
        if out:
            return out
    except Exception as exc:  # noqa: BLE001 -- error-path enrichment
        # degrade to the body-name fallback, surfacing WHY inline (the
        # original resolution failure must stay the headline)
        names = getattr(getattr(robot, "data", None), "body_names", None) or []
        if names:
            return [{"prim": str(n), "collision": True} for n in names]
        raise RuntimeError(
            f"asset inventory discovery failed ({exc!r}) and no body-name "
            "fallback is available") from exc
    names = getattr(getattr(robot, "data", None), "body_names", None) or []
    return [{"prim": str(n), "collision": True} for n in names]


InventoryFn = Callable[[], list[dict]]


def resolve_bucket_names(expected: Sequence[str], robot_bodies: list[str],
                         spec, inventory_fn: Optional[InventoryFn] = None,
                         backend: str = "isaaclab") -> list[str]:
    """General-group bucket resolution with the candidate-discovery refusal.

    Fullmatches the bucket's body-name patterns against the robot's body
    list (order preserved).  An empty match raises
    :class:`ContactCandidateError` carrying the discovered inventory --
    never a bare "matched nothing".
    """
    compiled = [re.compile(p) for p in expected]
    names = [n for n in robot_bodies
             if any(c.fullmatch(n) for c in compiled)]
    if names:
        return names
    inventory = (inventory_fn or (lambda: []))()
    raise contact_candidate_error(spec.name, "sensor_bodies",
                                  list(expected), inventory, backend,
                                  reason="matched no robot body")


def resolve_entry_name(entry: str, robot_bodies: list[str], spec,
                       index: int, inventory_fn: Optional[InventoryFn] = None,
                       backend: str = "isaaclab") -> str:
    """Pairwise-entry resolution (exactly one body) with the same refusal."""
    _, name = split_entry_scope(entry)
    compiled = re.compile(name)
    names = [n for n in robot_bodies if compiled.fullmatch(n)]
    if len(names) == 1:
        return names[0]
    inventory = (inventory_fn or (lambda: []))()
    reason = (f"entry {index} matched {len(names)} robot bodies"
              if names else
              f"entry {index} ({name!r}) matched no robot body")
    raise contact_candidate_error(spec.name, "sensor_bodies", [name],
                                  inventory, backend, reason=reason)


def split_entry_scope(entry: str) -> tuple[Optional[str], str]:
    """``<scope>/<name>`` -> (scope, name); mirrors the backend splitter."""
    if "/" in entry:
        scope, name = entry.rsplit("/", 1)
        return scope, name
    return None, entry
