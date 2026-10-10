# Executed INSIDE Blender via MCP.
# Debug overlay drawn with GPU draw handlers: creates NO objects or data-blocks, is never
# saved into the .blend file and disappears on `clear` or when Blender restarts.
#
# Contract: main({"action": "show" | "clear" | "status", "spec": {...}}) -> dict
# spec = {"lines": [[p0, p1, rgba], ...], "points": [[p, rgba, size], ...], "labels": [[p, text, rgba], ...]}

import bpy

_KEY = "animagent_overlay"


def _redraw():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _clear():
    state = bpy.app.driver_namespace.pop(_KEY, None)
    removed = 0
    if state:
        for handle, region_type in state.get("handles", []):
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, region_type)
                removed += 1
            except (ValueError, RuntimeError):
                pass
    _redraw()
    return removed


def _show(spec):
    import blf
    import gpu
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from gpu_extras.batch import batch_for_shader

    _clear()
    lines = spec.get("lines", [])
    points = spec.get("points", [])
    labels = spec.get("labels", [])

    line_shader = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
    point_shader = gpu.shader.from_builtin("FLAT_COLOR")
    line_pos, line_col = [], []
    for p0, p1, col in lines:
        line_pos += [tuple(p0), tuple(p1)]
        line_col += [tuple(col), tuple(col)]
    line_batch = batch_for_shader(line_shader, "LINES", {"pos": line_pos, "color": line_col}) if line_pos else None
    by_size = {}
    for p, col, size in points:
        by_size.setdefault(float(size), ([], []))
        by_size[float(size)][0].append(tuple(p))
        by_size[float(size)][1].append(tuple(col))
    point_batches = [(s, batch_for_shader(point_shader, "POINTS", {"pos": pos, "color": cols}))
                     for s, (pos, cols) in by_size.items()]
    line_width = float(spec.get("line_width", 2.5))

    def draw_3d():
        region = bpy.context.region
        gpu.state.depth_test_set("NONE")
        gpu.state.blend_set("ALPHA")
        if line_batch is not None:
            line_shader.bind()
            line_shader.uniform_float("viewportSize", (region.width, region.height))
            line_shader.uniform_float("lineWidth", line_width)
            line_batch.draw(line_shader)
        for size, batch in point_batches:
            gpu.state.point_size_set(size)
            point_shader.bind()
            batch.draw(point_shader)
        gpu.state.point_size_set(1.0)
        gpu.state.blend_set("NONE")

    legend = spec.get("legend")
    rect_shader = gpu.shader.from_builtin("UNIFORM_COLOR")

    def draw_2d():
        region = bpy.context.region
        rv3d = bpy.context.region_data
        if rv3d is None:
            return
        font = 0
        blf.enable(font, blf.SHADOW)
        blf.shadow(font, 3, 0.0, 0.0, 0.0, 1.0)
        blf.shadow_offset(font, 1, -1)
        blf.size(font, float(spec.get("label_size", 12)))
        for p, text, col in labels:
            co = location_3d_to_region_2d(region, rv3d, p)
            if co is None:
                continue
            blf.color(font, *col)
            blf.position(font, co[0] + 6, co[1] + 4, 0)
            blf.draw(font, text)
        if legend:
            size = float(legend.get("size", 13))
            items = legend.get("items", [])
            line_h = size * 1.45
            width = 12 + max([blf.dimensions(font, t)[0] for t, _ in items] + [blf.dimensions(font, legend.get("title", ""))[0]]) + 30
            height = line_h * (len(items) + 1) + 12
            x0, y1 = 12, region.height - 40
            gpu.state.blend_set("ALPHA")
            rect_shader.bind()
            rect_shader.uniform_float("color", (0.0, 0.0, 0.0, 0.72))
            batch_for_shader(rect_shader, "TRI_FAN", {"pos": [(x0, y1), (x0 + width, y1), (x0 + width, y1 - height), (x0, y1 - height)]}).draw(rect_shader)
            gpu.state.blend_set("NONE")
            blf.size(font, size)
            blf.color(font, 1, 1, 1, 1)
            blf.position(font, x0 + 8, y1 - line_h, 0)
            blf.draw(font, legend.get("title", ""))
            for i, (text, col) in enumerate(items):
                y = y1 - line_h * (i + 2)
                rect_shader.bind()
                rect_shader.uniform_float("color", tuple(col))
                batch_for_shader(rect_shader, "TRI_FAN", {"pos": [(x0 + 8, y), (x0 + 22, y), (x0 + 22, y + size * 0.8), (x0 + 8, y + size * 0.8)]}).draw(rect_shader)
                blf.color(font, 1, 1, 1, 1)
                blf.position(font, x0 + 30, y, 0)
                blf.draw(font, text)
        blf.disable(font, blf.SHADOW)

    h3 = bpy.types.SpaceView3D.draw_handler_add(draw_3d, (), "WINDOW", "POST_VIEW")
    h2 = bpy.types.SpaceView3D.draw_handler_add(draw_2d, (), "WINDOW", "POST_PIXEL")
    bpy.app.driver_namespace[_KEY] = {"handles": [(h3, "WINDOW"), (h2, "WINDOW")], "title": spec.get("title", "")}
    _redraw()
    return {"lines": len(lines), "points": len(points), "labels": len(labels)}


def main(args):
    action = args.get("action", "status")
    if bpy.app.background:
        return {"status": "unavailable", "reason": "background mode has no viewport"}
    if action == "clear":
        return {"status": "cleared", "handlers_removed": _clear()}
    if action == "show":
        return {"status": "shown", **_show(args.get("spec", {}))}
    state = bpy.app.driver_namespace.get(_KEY)
    return {"status": "active" if state else "inactive", "title": state.get("title") if state else None}
