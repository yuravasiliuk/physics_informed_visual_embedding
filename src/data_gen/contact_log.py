"""PhysX contact-event logging — exact impulses at physics rate.

WHY THIS IS LOAD-BEARING (material_oracle, 2026-07-27). Restitution is written
into a trajectory by ONE event, the impact, and the corpus measurement of it was
destroyed by sampling: state is logged per RENDERED frame, so gravity injects
`g·Δt = 9.81/30 = 0.327 m/s` into vz between samples while the median predicted
rebound was `e·|vz_in| = 0.178 m/s` — *below the floor*. Only ~34% of bounces
survived to the next sample and `corr(e_est, e_true)` collapsed to +0.08.

Contact reports fix this at the source, twice over:
  1. They fire at **physics rate** (60 Hz here, 2× the render rate), and
  2. the impulse is the **solver's own**, not a finite difference of sampled
     velocities — so it is immune to the sampling floor entirely.

They also supply the thing event-anchored targets cannot be built without: the
**event timestamps**. The oracle showed restitution decodable at R² +0.217 with
±4-frame windows around the impact vs *negative* R² from clip-level features, so
knowing *when* contact happened is not a nicety, it is the target's definition.

And they open mass/density ratio: for a two-body collision the pair impulses are
equal and opposite, so `m₁·Δv₁ = −m₂·Δv₂` reads the ratio exactly (R8-safe: the
volume ratio is visible but the density ratio is not).

TRUST GATE (verify_contacts.py, no Isaac): for an isolated single-body contact
the logged impulse must satisfy

    J ≈ m·Δv − m·g·Δt          (gravity's contribution over the step removed)

⚠ API RISK: the contact-report subscription below is the documented PhysX path
but is the one part of this module that cannot be checked off-Isaac. If a first
run produces zero events, look here first — threshold too high, the API not
applied before `world.reset()`, or a changed `omni.physx` entry point.
"""

import numpy as np

# omni imports are deferred into __init__ so this module can be imported (and
# its pure helpers unit-tested) without Isaac — the same discipline
# camera_paths/camera_motion follow.

GROUND_NAME = "ground"
SCENERY_NAME = "scenery"


def classify_actor(path, name_by_path):
    """Prim path -> logical body name. Unknown prims fall back to ground/scenery
    so a contact is never silently dropped for lack of a mapping."""
    if path in name_by_path:
        return name_by_path[path]
    low = path.lower()
    if "ground" in low or "floor" in low or "plane" in low:
        return GROUND_NAME
    return SCENERY_NAME


class ContactLogger:
    """Collects PhysX contact events, tagged with the render frame they fall in.

    Usage inside the capture loop:

        logger = ContactLogger(bodies, config)      # AFTER world.reset()
        for t in range(frame_count):
            logger.begin_frame(t)
            world.step(render=True)
            ...
        events = logger.drain(); logger.close()

    One physics step may produce several events; several physics steps fall
    inside one rendered frame. Every event therefore carries both `frame` (the
    rendered frame it belongs to — what windows are anchored on) and `substep`
    (its order within that frame), so nothing is lost to the render rate.
    """

    def __init__(self, bodies, config, threshold=0.0):
        from pxr import PhysxSchema
        import omni.physx
        from omni.isaac.core.utils.prims import get_prim_at_path

        self._physx = omni.physx.get_physx_simulation_interface()
        self.name_by_path = {}
        for body in bodies:
            path = str(body.prim_path)
            self.name_by_path[path] = str(body.name)
            prim = get_prim_at_path(path)
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            # threshold 0 = report every contact. Filtering here would silently
            # drop the weak impacts whose loss we are trying to fix.
            api.CreateThresholdAttr().Set(float(threshold))

        self._events = []
        self._frame = -1
        self._substep = 0
        self._sub = self._physx.subscribe_contact_report_events(self._on_contact)

    # -- callback ---------------------------------------------------------
    def _on_contact(self, contact_headers, contact_data):
        from pxr import PhysicsSchemaTools
        for header in contact_headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            lo = int(header.contact_data_offset)
            hi = lo + int(header.num_contact_data)
            if hi <= lo:
                continue
            # Sum the impulse over the pair's contact points (that is the net
            # pair impulse, which is what momentum bookkeeping needs); average
            # position/normal for a representative contact geometry.
            imp = np.zeros(3)
            pos = np.zeros(3)
            nrm = np.zeros(3)
            n = 0
            for i in range(lo, hi):
                cd = contact_data[i]
                imp += np.asarray([cd.impulse[0], cd.impulse[1], cd.impulse[2]], float)
                pos += np.asarray([cd.position[0], cd.position[1], cd.position[2]], float)
                nrm += np.asarray([cd.normal[0], cd.normal[1], cd.normal[2]], float)
                n += 1
            if n == 0:
                continue
            nn = np.linalg.norm(nrm)
            self._events.append({
                "frame": int(self._frame),
                "substep": int(self._substep),
                "body_a": classify_actor(a0, self.name_by_path),
                "body_b": classify_actor(a1, self.name_by_path),
                "prim_a": a0,
                "prim_b": a1,
                "impulse_world": imp.tolist(),          # N·s, summed over points
                "point_world": (pos / n).tolist(),
                "normal_world": (nrm / nn).tolist() if nn > 1e-12 else [0.0, 0.0, 1.0],
                "n_points": int(n),
                "event": int(getattr(header, "type", -1)),   # FOUND/PERSIST/LOST
            })
        self._substep += 1

    # -- loop hooks -------------------------------------------------------
    def begin_frame(self, frame_idx):
        """Tag subsequent events with this rendered frame. Call BEFORE the step."""
        self._frame = int(frame_idx)
        self._substep = 0

    def drain(self):
        out, self._events = self._events, []
        return out

    def close(self):
        try:
            if self._sub is not None:
                self._sub.unsubscribe()
        except Exception:
            pass
        self._sub = None


# ---------------------------------------------------------------------------
# Pure helpers — no Isaac, unit-testable, shared with verify_contacts.py
# ---------------------------------------------------------------------------

def to_camera_frame(events, R_camera_world, camera_pos_world):
    """Add camera-frame copies of the world-frame vectors (repo invariant:
    labels live in camera frame; world stays recoverable via the extrinsics).

    `R_camera_world`/`camera_pos_world` may be per-event (dynamic camera) or a
    single value (static). Vectors rotate; the contact POINT also translates.
    """
    R = np.asarray(R_camera_world, dtype=np.float64)
    p0 = np.asarray(camera_pos_world, dtype=np.float64)
    out = []
    for e in events:
        e = dict(e)
        e["impulse_camera"] = (R @ np.asarray(e["impulse_world"])).tolist()
        e["normal_camera"] = (R @ np.asarray(e["normal_world"])).tolist()
        e["point_camera"] = (R @ (np.asarray(e["point_world"]) - p0)).tolist()
        out.append(e)
    return out


def net_impulse_per_body(events, body_name):
    """Sum of impulses acting ON `body_name` within an event list.

    Sign convention: PhysX reports the impulse for the (actor0, actor1) pair;
    the reaction on actor1 is its negation (Newton's third law). Getting this
    backwards is the classic silent error, so it is centralized here and gated
    by verify_contacts.py's momentum identity.
    """
    total = np.zeros(3)
    for e in events:
        j = np.asarray(e["impulse_world"], dtype=np.float64)
        if e["body_a"] == body_name:
            total += j
        elif e["body_b"] == body_name:
            total -= j
    return total


def restitution_from_impulse(j_normal, mass, v_in_normal):
    """Closed-form e from a normal impulse and the approach speed.

    For a body bouncing off immovable ground, J = m(1+e)|v_in·n| ⇒
        e = J / (m·|v_in·n|) − 1
    Sampling-independent: J comes from the solver, not from differencing
    rendered frames — which is the whole reason this module exists.
    """
    denom = float(mass) * abs(float(v_in_normal))
    if denom < 1e-9:
        return float("nan")
    return float(j_normal) / denom - 1.0
