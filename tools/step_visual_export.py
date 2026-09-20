"""STEP to visual-mesh conversion via the prebuilt ``superdex_mesh_cli`` helper.

Requires the project environment (Python >= 3.12 with the ``superdex-mesh-cli``
wheel installed); run from the project root as ``uv run tools/step_visual_export.py``.


This is the CAD-to-visual-mesh path that is otherwise only reachable through
``superdex_cad_exporter`` (``MeshExporter.Export`` / ``GeometryOp.ExportStepVisual``):
the STEP file is loaded once, tessellated with analytic normals and CAD colors
preserved, and written out as one or more render-ready files (GLB/glTF/OBJ/STL).

Nothing is built from source. The script shells out to the helper executable
shipped inside the installed ``superdex-mesh-cli`` wheel
(``superdex_mesh_cli/_native/``), speaking the same framed binary protocol as
the C# exporter and the C++ ``mochi_mesh`` client. Only the Python standard
library is used.

Run from the project root so ``uv`` resolves the project environment::

    uv run tools/step_visual_export.py input.stp -o out.glb
    uv run tools/step_visual_export.py input.stp -o out.glb -o out.obj --backend delabella
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

FRAME_MAGIC = 0x53445832
PROTOCOL_VERSION = 3
OP_EXPORT_STEP_VISUAL = 10

FORMATS = {"glb": 0, "gltf": 1, "obj": 2, "stl": 3}
BACKENDS = {"delabella": 0, "isotropic": 1}
EDGE_SAMPLINGS = {"uniform": 0, "adaptive": 1}
STATUSES = ("Written", "WrittenPartial", "Failed")


class MeshCliError(RuntimeError):
    """The helper is missing, crashed, or reported a conversion failure."""


def find_helper(explicit: str | None = None) -> Path:
    """Locate the prebuilt helper without building anything.

    Mirrors ``FindMeshCliPath`` / ``MeshCliRunner.FindExecutable``: an explicit
    path (or ``SUPERDEX_MESH_CLI_PATH``) wins, otherwise the executable sitting
    in ``_native/`` beside the installed ``superdex_mesh_cli`` package is used.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if env := os.environ.get("SUPERDEX_MESH_CLI_PATH"):
        candidates.append(Path(env))
    exe_name = "superdex_mesh_cli.exe" if os.name == "nt" else "superdex_mesh_cli"
    try:
        import superdex_mesh_cli

        candidates.append(Path(superdex_mesh_cli.__file__).parent / "_native" / exe_name)
    except ImportError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates) or "no candidates"
    raise MeshCliError(
        f"superdex_mesh_cli helper not found ({searched}). Install the "
        "superdex-mesh-cli distribution or set SUPERDEX_MESH_CLI_PATH."
    )


def _write_byte_array(buf: bytearray, data: bytes) -> None:
    buf += struct.pack("=Q", len(data))
    buf += data


def build_payload(
    step_path: str,
    outputs: list[tuple[int, str]],
    *,
    backend: int,
    linear_deflection: float,
    angular_deflection: float,
    target_edge_length: float,
    target_edge_length_fraction: float,
    edge_sampling: int,
    scale: float,
    allow_partial_failure: bool,
    rgba_material_names: bool,
) -> bytes:
    """Encode the ExportStepVisual payload field by field.

    Layout mirrors ``WriteStepVisualExportParams`` (declaration order) followed
    by ``WriteVisualExportOutputs``; see ``mochi_mesh_cli_encoding.cpp``.
    """
    payload = bytearray()
    _write_byte_array(payload, step_path.encode("utf-8"))
    payload += struct.pack("=I", backend)
    payload += struct.pack("=d", linear_deflection)
    payload += struct.pack("=d", angular_deflection)
    payload += struct.pack("=d", target_edge_length)
    payload += struct.pack("=d", target_edge_length_fraction)
    payload += struct.pack("=I", edge_sampling)
    payload += struct.pack("=d", scale)
    payload += struct.pack("=I", 1 if allow_partial_failure else 0)
    payload += struct.pack("=I", 1 if rgba_material_names else 0)
    payload += struct.pack("=Q", len(outputs))
    for fmt, path in outputs:
        payload += struct.pack("=I", fmt)
        _write_byte_array(payload, path.encode("utf-8"))
    return bytes(payload)


def invoke_helper(helper: Path, payload: bytes, timeout: float) -> list[int]:
    """Send one ExportStepVisual request; return one status per output."""
    request = (
        struct.pack("=IIIQ", FRAME_MAGIC, PROTOCOL_VERSION, OP_EXPORT_STEP_VISUAL, len(payload))
        + payload
    )
    try:
        proc = subprocess.run(
            [str(helper)],
            input=request,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: helper diagnostics + OCCT chatter go there
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MeshCliError(f"failed to start {helper}: {exc}") from exc
    if proc.returncode != 0:
        raise MeshCliError(f"superdex_mesh_cli exited with status {proc.returncode}.")
    frame = proc.stdout
    header_size = struct.calcsize("=IIQ")
    if len(frame) < header_size:
        raise MeshCliError("superdex_mesh_cli returned a truncated or empty response.")
    magic, status, payload_len = struct.unpack_from("=IIQ", frame)
    if magic != FRAME_MAGIC or payload_len != len(frame) - header_size:
        raise MeshCliError("superdex_mesh_cli response has a bad magic or length.")
    body = frame[header_size:]
    if status != 0:
        message = body.decode("utf-8", errors="replace") or (
            "superdex_mesh_cli reported an error processing the request."
        )
        raise MeshCliError(message)
    if len(body) < 8:
        raise MeshCliError("superdex_mesh_cli response is missing its status array.")
    (count,) = struct.unpack_from("=Q", body)
    if count != len(body[8:]) // 4 or len(body[8:]) % 4:
        raise MeshCliError("superdex_mesh_cli returned a malformed status array.")
    statuses = list(struct.unpack(f"={count}I", body[8:]))
    if any(s >= len(STATUSES) for s in statuses):
        raise MeshCliError(f"superdex_mesh_cli returned unknown export status {statuses}.")
    return statuses


def infer_format(path: str) -> int:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in FORMATS:
        raise MeshCliError(
            f"cannot infer format from '{path}'; use --format with one of "
            f"{', '.join(FORMATS)}."
        )
    return FORMATS[ext]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a STEP CAD file to render-ready mesh file(s) "
        "using the prebuilt superdex_mesh_cli helper."
    )
    parser.add_argument("step", help="Input STEP file (.step/.stp)")
    parser.add_argument(
        "-o",
        "--output",
        action="append",
        required=True,
        metavar="FILE",
        help="Output file to write (repeatable; the STEP is loaded and tessellated "
        "once for all outputs). Format defaults to the file extension.",
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=sorted(FORMATS),
        metavar="FMT",
        help="Format for the corresponding --output (repeatable, in order).",
    )
    parser.add_argument("--backend", default="isotropic", choices=sorted(BACKENDS))
    parser.add_argument("--linear-deflection", type=float, default=0.1)
    parser.add_argument("--angular-deflection", type=float, default=0.5)
    parser.add_argument("--target-edge-length", type=float, default=0.0)
    parser.add_argument("--target-edge-length-fraction", type=float, default=0.02)
    parser.add_argument("--edge-sampling", default="adaptive", choices=sorted(EDGE_SAMPLINGS))
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Multiplier applied to the millimetre-normalized STEP geometry "
        "(0.001 converts to metres).",
    )
    parser.add_argument(
        "--allow-partial-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a partial file when some faces fail to tessellate.",
    )
    parser.add_argument(
        "--rgba-material-names",
        action="store_true",
        help="Rename each glTF material to the RRGGBBAA hex of its base color.",
    )
    parser.add_argument(
        "--mesh-cli",
        default=None,
        help="Explicit helper path (also: SUPERDEX_MESH_CLI_PATH).",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.format and len(args.format) != len(args.output):
        print(
            "error: --format count must match --output count.", file=sys.stderr
        )
        return 2
    outputs = [
        (FORMATS[fmt] if fmt else infer_format(path), path)
        for path, fmt in zip(
            args.output, args.format or [None] * len(args.output)
        )
    ]
    try:
        helper = find_helper(args.mesh_cli)
        statuses = invoke_helper(
            helper,
            build_payload(
                args.step,
                outputs,
                backend=BACKENDS[args.backend],
                linear_deflection=args.linear_deflection,
                angular_deflection=args.angular_deflection,
                target_edge_length=args.target_edge_length,
                target_edge_length_fraction=args.target_edge_length_fraction,
                edge_sampling=EDGE_SAMPLINGS[args.edge_sampling],
                scale=args.scale,
                allow_partial_failure=args.allow_partial_failure,
                rgba_material_names=args.rgba_material_names,
            ),
            args.timeout,
        )
    except MeshCliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    failed = 0
    for (_, path), status in zip(outputs, statuses):
        print(f"{path}: {STATUSES[status]}")
        failed += status == 2
    if len(statuses) != len(outputs):
        print(
            f"error: helper returned {len(statuses)} status(es) for "
            f"{len(outputs)} output(s).",
            file=sys.stderr,
        )
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
