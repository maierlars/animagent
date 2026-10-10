# Executed INSIDE Blender via MCP. Temporarily frames the 3D viewport for screenshots and
# restores the user's previous view afterwards. Only UI view state is touched (no scene data).
#
# Contract: main({"action": "frame" | "restore", "view": "front"|"back"|"left"|"right"|"top"|"persp",
#                 "center": [x,y,z], "size": float, "forward": [..], "up": [..], "left": [..]})

import bpy
from mathutils import Matrix, Vector

_KEY = "animagent_saved_view"


def _view3d():
    best = None
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D" and (best is None or area.width * area.height > best[1].width * best[1].height):
                best = (win, area)
    if best is None:
        raise RuntimeError("No 3D viewport found")
    win, area = best
    space = area.spaces.active
    return win, area, space, space.region_3d


def _force_redraw(win, area):
    area.tag_redraw()
    try:
        with bpy.context.temp_override(window=win, area=area):
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass


def main(args):
    if bpy.app.background:
        return {"status": "unavailable", "reason": "background mode has no viewport"}
    win, area, space, rv3d = _view3d()
    if args.get("action") == "restore":
        saved = bpy.app.driver_namespace.pop(_KEY, None)
        if saved is None:
            return {"status": "nothing_to_restore"}
        rv3d.view_location = Vector(saved["location"])
        rv3d.view_rotation = saved["rotation"]
        rv3d.view_distance = saved["distance"]
        rv3d.view_perspective = saved["perspective"]
        _force_redraw(win, area)
        return {"status": "restored"}

    if _KEY not in bpy.app.driver_namespace:
        bpy.app.driver_namespace[_KEY] = {
            "location": tuple(rv3d.view_location), "rotation": rv3d.view_rotation.copy(),
            "distance": rv3d.view_distance, "perspective": rv3d.view_perspective,
        }
    fwd, up, left = Vector(args["forward"]), Vector(args["up"]), Vector(args["left"])
    view = args.get("view", "front")
    # Direction the camera looks along (from eye to subject).
    look = {
        "front": -fwd, "back": fwd, "left": -left, "right": left, "top": -up,
        "persp": (-fwd - 0.8 * left - 0.5 * up).normalized(),
    }[view]
    cam_up = fwd if view == "top" else up
    z = (-look).normalized()
    x = cam_up.cross(z).normalized()
    y = z.cross(x).normalized()
    rv3d.view_rotation = Matrix((x, y, z)).transposed().to_quaternion()
    rv3d.view_location = Vector(args["center"])
    rv3d.view_distance = float(args.get("size", 2.0)) * 1.25
    rv3d.view_perspective = "PERSP" if view == "persp" else "ORTHO"
    _force_redraw(win, area)
    return {"status": "framed", "view": view, "area": [area.width, area.height]}
