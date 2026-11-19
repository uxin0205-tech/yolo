"""Stable variant CLI with the exact tiled XNOR backend installed first."""

from __future__ import annotations

from . import _variant_cli_impl as _impl
from .source import SourceBundle
from .xnor import XNORExecutionConfig, install_xnor_backend

_original_pose = _impl._pose
_original_joint_smoke = _impl._joint_smoke


def _install_for(workspace) -> None:
    source = SourceBundle(
        workspace.source_bundle,
        architecture=workspace.architecture,
    )
    source.verify_manifest()
    source.verify_environment()
    source.activate_code()
    install_xnor_backend(XNORExecutionConfig(token_tile=32))


def _pose(workspace, args):
    _install_for(workspace)
    return _original_pose(workspace, args)


def _joint_smoke(workspace, args):
    _install_for(workspace)
    return _original_joint_smoke(workspace, args)


_impl._pose = _pose
_impl._joint_smoke = _joint_smoke
main = _impl.main

__all__ = ("main",)

