# Executed INSIDE Blender via MCP. Read-only. Diagnostic information for `animagent doctor`.

import sys

import addon_utils
import bpy


def main(args):
    armatures = []
    for ob in bpy.data.objects:
        if ob.type != "ARMATURE":
            continue
        names = [b.name for b in ob.data.bones]
        armatures.append({
            "name": ob.name,
            "bone_count": len(names),
            "rig_id": ob.data.get("rig_id"),
            "org_bones": sum(n.startswith("ORG-") for n in names),
            "def_bones": sum(n.startswith("DEF-") for n in names),
            "deform_bones": sum(b.use_deform for b in ob.data.bones),
            "is_metarig": any(getattr(pb, "rigify_type", "") for pb in ob.pose.bones),
            "visible": bool(ob.visible_get()),
            "deformed_meshes": sorted(
                m.name for m in bpy.data.objects
                if m.type == "MESH" and any(md.type == "ARMATURE" and md.object is ob for md in m.modifiers)
            ),
        })
    rigify = addon_utils.check("rigify")
    return {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
        "python_version": sys.version.split()[0],
        "background": bool(bpy.app.background),
        "file_path": bpy.data.filepath or None,
        "is_dirty": bool(bpy.data.is_dirty),
        "scene": bpy.context.scene.name,
        "frame_current": bpy.context.scene.frame_current,
        "mode": bpy.context.mode,
        "object_count": len(bpy.data.objects),
        "armatures": armatures,
        "rigify_addon": {"loaded_default": bool(rigify[0]), "loaded_state": bool(rigify[1])},
        "numpy": __import__("numpy").__version__,
    }
