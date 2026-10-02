"""State adapter for IsaacSim with quaternion format conversion.

This module provides a lightweight adapter that handles object state access
with automatic quaternion format conversion between IsaacSim's native wxyz
format and holosoma's standard xyzw format.
"""

from __future__ import annotations

import torch
from .state_utils import fullstate_wxyz_to_xyzw, fullstate_xyzw_to_wxyz


class IsaacSimStateAdapter:
    """Lightweight adapter for IsaacSim state access with quaternion conversion.

    This adapter provides a clean interface for object state management while
    handling the quaternion format conversion between IsaacSim (wxyz) and
    holosoma (xyzw) automatically.

    Parameters
    ----------
    device : torch.device
        Device for tensor operations
    object_registry : ObjectRegistry
        Registry for object index resolution
    scene : InteractiveScene
        IsaacLab scene containing rigid objects
    robot_states : RootStatesProxy
        Robot states proxy (already handles wxyz->xyzw conversion)
    """

    def __init__(self, device: torch.device, object_registry, scene, robot, robot_states):
        self.device = device
        self._object_registry = object_registry
        self._scene = scene
        self._robot = robot
        self._robot_states = robot_states
        self._objects_dirty = False

    def resolve_indices(self, indices: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Resolve flat indices to (object_name, env_ids) pairs.

        Parameters
        ----------
        indices : torch.Tensor
            Flat tensor indices to resolve

        Returns
        -------
        list[tuple[str, torch.Tensor]]
            List of (object_name, env_ids) pairs
        """
        return self._object_registry.resolve_indices(indices)

    def get_object_states(self, obj_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states with automatic wxyz->xyzw conversion.

        Parameters
        ----------
        obj_name : str
            Name of the object to get states for
        env_ids : torch.Tensor
            Environment IDs to query

        Returns
        -------
        torch.Tensor
            Object states in xyzw format, shape [len(env_ids), 13]

        Raises
        ------
        ValueError
            If object type is unknown
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            # Robot states already converted to xyzw via RootStatesProxy
            return self._robot_states[env_ids]

        elif obj_type == "individual":
            # Individual rigid objects - convert from wxyz to xyzw
            rigid_object = self._scene.rigid_objects[obj_name]
            raw_states = rigid_object.data.root_state_w[env_ids]
            return fullstate_wxyz_to_xyzw(raw_states)

        elif obj_type == "scene":
            # Scene collection objects - convert from wxyz to xyzw
            scene_collection = self._scene.rigid_objects["usd_scene_objects"]
            object_index = self._object_registry.get_scene_position(obj_name)
            raw_states = scene_collection.data.object_state_w[env_ids, object_index]
            return fullstate_wxyz_to_xyzw(raw_states)

        else:
            raise ValueError(f"Unknown object type '{obj_type}' for object '{obj_name}'")

    @staticmethod
    def _normalize_names(names: str | list[str]) -> list[str]:
        normalized = [names] if isinstance(names, str) else list(names)
        if any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError("actor names must be non-empty strings")
        return normalized

    def get_named_object_states(
        self, names: str | list[str], env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Read public actor states in exact caller name/env order and xyzw format."""
        normalized = self._normalize_names(names)
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if not normalized:
            return torch.empty(
                (0, 13), device=self.device, dtype=torch.float32
            )
        rows = [self.get_object_states(name, env_ids) for name in normalized]
        result = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        expected = (len(normalized) * env_ids.numel(), 13)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"named actor state shape {tuple(result.shape)} != {expected}"
            )
        return result

    def write_named_object_states(
        self, names: str | list[str], states: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        """Write public actor states in exact caller name/env order from xyzw format."""
        normalized = self._normalize_names(names)
        if not isinstance(env_ids, torch.Tensor) or env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        expected = (len(normalized) * env_ids.numel(), 13)
        if not isinstance(states, torch.Tensor) or tuple(states.shape) != expected:
            shape = tuple(states.shape) if isinstance(states, torch.Tensor) else None
            raise ValueError(f"named actor states shape {shape} != {expected}")
        stride = env_ids.numel()
        for index, name in enumerate(normalized):
            self.write_object_states(
                name, states[index * stride : (index + 1) * stride], env_ids
            )

    def write_object_states(self, obj_name: str, states: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write object states with automatic xyzw->wxyz conversion.

        Parameters
        ----------
        obj_name : str
            Name of the object to update
        states : torch.Tensor
            New states in xyzw format, shape [len(env_ids), 13]
        env_ids : torch.Tensor
            Environment IDs to update
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            # Write robot states
            # Converts manually and skips RootStatesProxy due to unified interface via AllRootStatesProxy
            # NOTE: Intentionally does NOT apply env origins offsets for backwards compatibilty
            #       with existing robot root state setters
            converted_states = fullstate_xyzw_to_wxyz(states)
            self._robot.write_root_pose_to_sim(converted_states[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(converted_states[:, 7:], env_ids)

        elif obj_type == "individual":
            # Individual rigid objects - convert to wxyz and apply env origins
            rigid_object = self._scene.rigid_objects[obj_name]
            converted_states = fullstate_xyzw_to_wxyz(states)
            # WBT applies environment origins to these states.


            rigid_object.write_root_pose_to_sim(converted_states[:, :7], env_ids)
            rigid_object.write_root_velocity_to_sim(converted_states[:, 7:], env_ids)

        elif obj_type == "scene":
            # Scene collection objects - convert to wxyz and write to collection
            scene_collection = self._scene.rigid_objects["usd_scene_objects"]
            object_index = self._object_registry.get_scene_position(obj_name)

            converted_states = fullstate_xyzw_to_wxyz(states)
            converted_states[:, 0:3] += self._scene.env_origins[env_ids]  # Apply environment origins

            # Update the collection's state tensor
            current_states = scene_collection.data.object_state_w[env_ids].clone()
            current_states[:, object_index] = converted_states
            scene_collection.write_object_state_to_sim(current_states, env_ids)

        else:
            raise ValueError(f"Unknown object type '{obj_type}' for object '{obj_name}'")

        # Mark states as dirty for batch synchronization
        self.mark_dirty()

    def is_dirty(self) -> bool:
        """Check if states have been modified and need synchronization.

        Returns
        -------
        bool
            True if states are dirty and need sync
        """
        return self._objects_dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag after synchronization."""
        self._objects_dirty = False

    def mark_dirty(self) -> None:
        """Explicitly mark states as dirty."""
        self._objects_dirty = True
