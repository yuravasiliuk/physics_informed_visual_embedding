"""Class-label semantics for the segmentation annotator.

Step 5.8 of `plans/data-gen/scene-diversification.md` configures Replicator's
`instance_segmentation_fast` annotator to group prims by `class` label
rather than per-prim instance. This module is the single place where the
labels get authored, so the convention is uniform across:

    - primitive dynamic bodies  (body_factory.build_random_bodies)  → label = body name
    - primitive scenery         (static_geometry.make_*)            → label = SCENERY_LABEL
    - asset-loaded dynamic body (asset_loader.load_dynamic_asset)   → label = body name
    - asset-loaded scenery      (asset_loader.load_static_asset)    → label = SCENERY_LABEL (caller's choice)

Without uniform labels, step 5.8's annotator switch would group asset bodies
correctly but leave primitive bodies un-grouped, producing inconsistent seg
IDs across clips.
"""
from pxr import Semantics

SCENERY_LABEL = "scenery"


def apply_class_label(prim, label: str) -> None:
    """Attach a `class` semantic with `label` to `prim`.

    Multi-applied schema instance name is "Semantics" — the convention Isaac's
    own `omni.isaac.core.utils.semantics` helper uses by default, so prims
    tagged by either path interoperate.
    """
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(label)
