"""legacy_study: feasible role allocation, coordinator and paired evaluation."""

from dart_marl.roles.allocator import AllocationConfig, allocate_roles
from dart_marl.roles.coordinator import RoleCoordinator

__all__ = ["AllocationConfig", "allocate_roles", "RoleCoordinator"]
