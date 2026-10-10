from animagent.analysis.skeleton import build_skeleton
from tests.fakes.synthetic import humanoid_extraction, rigify_like_extraction


def test_deform_strategy_uses_deform_bones() -> None:
    sk = build_skeleton(humanoid_extraction())
    assert sk.strategy == "deform"
    assert sk.roots == ["hips"]
    bm = sk.bone_map()
    assert bm["shin.L"].parent == "thigh.L"
    assert bm["thigh.L"].side.value == "L"


def test_rigify_strategy_uses_org_bones_and_metarig_parents() -> None:
    ex = rigify_like_extraction()
    sk = build_skeleton(ex)
    assert sk.strategy == "rigify_org"
    assert all(b.name.startswith("ORG-") for b in sk.bones)
    bm = sk.bone_map()
    assert bm["ORG-hips"].parent is None
    assert bm["ORG-forearm.L"].parent == "ORG-upper_arm.L"
    assert bm["ORG-forearm.L"].parent_source == "metarig"
    # Deform bones are mapped onto structural bones by geometry.
    assert bm["ORG-forearm.L"].deform_bones == ["DEF-forearm.L"]


def test_rigify_strategy_without_metarig_matches_metarig_result() -> None:
    ex = rigify_like_extraction()
    with_meta = {b.name: b.parent for b in build_skeleton(ex).bones}
    ex.rigify.metarig_bones = []
    without = {b.name: b.parent for b in build_skeleton(ex).bones}
    assert with_meta == without


def test_geometric_repair_connects_detached_bone() -> None:
    ex = humanoid_extraction()
    # Detach the shin in the hierarchy; its head still coincides with the thigh's tail.
    b = ex.bone_map()["shin.L"]
    b.parent = None
    sk = build_skeleton(ex)
    assert sk.bone_map()["shin.L"].parent == "thigh.L"
    assert sk.bone_map()["shin.L"].parent_source == "geometry"
