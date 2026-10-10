"""Run an isolated background Blender with the official MCP add-on bridge for integration tests.

The official add-on provides a background bridge (`cli.cli_execute`); since the add-on may
not be enabled in the user's saved preferences, a tiny launcher enables it for this
process only (preferences are not saved) and starts the bridge on a free port.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LAUNCHER = r"""
import importlib, os, sys
import addon_utils, bpy
port = int(os.environ["ANIMAGENT_TEST_PORT"])
wanted = os.environ.get("ANIMAGENT_BLENDER_MCP_ADDON")
candidates = [wanted] if wanted else []
for mod in addon_utils.modules():
    path = getattr(mod, "__file__", "") or ""
    if os.path.exists(os.path.join(os.path.dirname(path), "mcp_to_blender_server.py")):
        candidates.append(mod.__name__)
if not candidates:
    print("ANIMAGENT: Blender MCP add-on not installed", flush=True)
    sys.exit(97)
addon_utils.enable(candidates[0], default_set=False)
addon_utils.enable("rigify", default_set=False)
cli = importlib.import_module(candidates[0] + ".cli")
print("ANIMAGENT: starting bridge", candidates[0], port, flush=True)
sys.exit(cli.cli_execute(["--port", str(port)]))
"""


def blender_binary() -> str | None:
    return os.environ.get("BLENDER_BIN") or shutil.which("blender")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


@contextmanager
def background_blender(blend_file: Path | None = None, timeout: float = 90.0) -> Iterator[int]:
    binary = blender_binary()
    if binary is None:
        raise RuntimeError("blender not found")
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="animagent_bl_") as tmp:
        launcher = Path(tmp) / "launch.py"
        launcher.write_text(LAUNCHER)
        log_path = Path(tmp) / "blender.log"
        cmd = [binary, "--background"]
        if blend_file is not None:
            cmd.append(str(blend_file))
        cmd += ["--online-mode", "--python", str(launcher)]
        env = {**os.environ, "ANIMAGENT_TEST_PORT": str(port)}
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Blender exited ({proc.returncode}):\n{log_path.read_text()[-3000:]}")
                try:
                    with socket.create_connection(("localhost", port), timeout=0.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"Blender bridge did not start:\n{log_path.read_text()[-3000:]}") from None
                    time.sleep(0.3)
            yield port
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# Builds a Rigify human with a skinned box-mesh body. Runs inside Blender (test setup only).
BUILD_FIXTURE = r"""
import bpy, bmesh
from mathutils import Vector
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.object.armature_human_metarig_add()
meta = bpy.context.active_object
meta.name = "metarig"
bpy.ops.pose.rigify_generate()
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.data.get("rig_id"))
rig.name = "Rig"
me = bpy.data.meshes.new("Body")
bm = bmesh.new()
owners = []
for b in rig.data.bones:
    if not b.use_deform:
        continue
    M = rig.matrix_world @ b.matrix_local
    r = max(0.012, b.length * 0.2)
    vs = [bm.verts.new(M @ Vector((x, y, z))) for y in (0.0, b.length) for x in (-r, r) for z in (-r, r)]
    for f in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        bm.faces.new([vs[i] for i in f])
    owners.append((b.name, len(vs)))
bm.to_mesh(me)
bm.free()
body = bpy.data.objects.new("Body", me)
bpy.context.scene.collection.objects.link(body)
start = 0
for name, n in owners:
    vg = body.vertex_groups.new(name=name)
    vg.add(list(range(start, start + n)), 1.0, "REPLACE")
    start += n
mod = body.modifiers.new("Armature", "ARMATURE")
mod.object = rig
pb = rig.pose.bones["torso"]
pb.location.z = 0.05
pb.keyframe_insert("location", frame=1)
pb.location.z = 0.0
pb.keyframe_insert("location", frame=10)
bpy.context.scene.frame_set(5)
bpy.ops.wm.save_as_mainfile(filepath=ARGS_PATH)
result = {"rig": rig.name, "bones": len(rig.data.bones), "verts": len(me.vertices)}
"""
