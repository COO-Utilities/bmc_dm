"""Driver for Boston Micromachines (BMC) deformable mirrors over the BMC DM-SDK

Each mirror is described by a config bundle: a TOML file with actuator_map.txt and the
flats beside it (see DMConfig.load). Pattern functions build unit surfaces on an
ActuatorMap; BmcDmController.poke() scales them onto a bias, by default the flat.

Hardware notes:
    * Commands are normalized to the 0 to 1 hardware range and clipped to the soft
      limits before they are sent.
    * The SDK cannot read the mirror back: get_actuator_data() returns its own copy of
      the last command, which is zeros right after connect. The controller saves every
      command as last_command.npy beside the config and loads it on connect instead.
    * connect() sends no commands. disconnect() sends none either, but the SDK's
      close_dm() (and its BmcDm destructor) zeroes and disables the DACs, so call
      restore() after connect() to put the saved shape back.
    * status_report() summarizes what was sent without calling the SDK.
    * Needs the BMC DM-SDK Python module ``bmc`` (installed under
      /opt/Boston Micromachines/lib/python3) and the DM's profile from
      BMC. Zernikes need poppy (the zernike extra).
"""

import threading
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Final, NoReturn, Self

import numpy as np
from hardware_device_base import HardwareDeviceBase
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type IndexArray = NDArray[np.intp]
type Pattern = Callable[..., FloatArray]

HARDWARE_MIN: Final = 0.0
HARDWARE_MAX: Final = 1.0
NO_ACTUATOR: Final = -1
ACTUATOR_MAP_FILE: Final = "actuator_map.txt"
LAST_COMMAND_FILE: Final = "last_command.npy"
DEFAULT_LOGFILE: Final = "bmc_dm"

# Expected type of every config key; [flats] takes any name with a file name
_CONFIG_SCHEMA: Final[Mapping[str, Mapping[str, type]]] = MappingProxyType(
    {
        "dm": {"name": str, "serial_number": str, "default_flat": str},
        "limits": {"minimum": float, "maximum": float},
        "flats": {},
        "sdk": {"profiles_path": str, "maps_path": str},
        "log": {"file": str},
    }
)

# KPIC's F as (row start, row stop, column start, column stop) on the 34-wide kilo-DM,
# where it is centered; other grids scale it so it stays centered
_F_REFERENCE_SIDE: Final = 34
_F_STROKES: Final = ((5, 29, 10, 13), (5, 8, 13, 25), (14, 17, 13, 25))

# flash_edge alternates every third edge actuator starting at index 0 and at index 1
_FLASH_EDGE_STEP: Final = 3

# Generate Zernike bases in power-of-two sizes (at least this many terms) so nearby
# Noll requests share one cached basis
_MIN_ZERNIKE_TERMS: Final = 16


class DMError(Exception):
    """Raised when the SDK reports a failure or the DM is not connected"""


class ConfigError(ValueError):
    """Raised when a config bundle or one of its files is invalid"""


def _read_only[T: np.generic](array: NDArray[T]) -> NDArray[T]:
    """Lock an array against in-place edits and return it"""
    array.flags.writeable = False
    return array


class ActuatorMap:
    """Square grid giving the command vector index of the actuator at each cell

    Cells without an actuator hold NO_ACTUATOR (-1), and every index 0 to N-1 appears
    once. All arrays are read-only.

    :raises ValueError: if the grid is not square or does not number 0 to N-1 once
    """

    def __init__(self, indices: ArrayLike) -> None:
        """Validate the grid and precompute the mask, vector order and edge actuators"""
        layout = np.array(indices, dtype=np.intp)
        if layout.ndim != 2 or layout.shape[0] != layout.shape[1]:
            raise ValueError(f"Actuator map must be a square grid, got shape {layout.shape}")
        active = np.sort(layout[layout != NO_ACTUATOR])
        if not active.size or not np.array_equal(active, np.arange(active.size)):
            raise ValueError(f"Actuator map must hold {NO_ACTUATOR} or each index 0 to N-1 once")
        self.indices = _read_only(layout)
        self.mask = _read_only(layout != NO_ACTUATOR)
        self.side = int(layout.shape[0])
        self.shape = (self.side, self.side)
        self.num_actuators = int(active.size)
        # Flat grid position of each actuator, in command vector order
        positions = np.flatnonzero(self.mask)
        self._cells = _read_only(positions[np.argsort(layout.ravel()[positions])])
        # Edge actuators are missing a neighbor above, below, left or right
        padded = np.pad(self.mask, 1)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        self.edge_actuators = _read_only(np.sort(layout[self.mask & ~interior]))

    @classmethod
    def circular(cls, side: int) -> Self:
        """Build BMC's circular kilo-DM layout, numbered row by row (648 at 28, 952 at 34)"""
        row, column = np.mgrid[:side, :side]
        center = (side - 1) / 2
        # The +1 makes the radius match BMC's active region
        inside = np.hypot(row - center, column - center) < (side + 1) / 2
        layout = np.full((side, side), NO_ACTUATOR)
        layout[inside] = np.arange(np.count_nonzero(inside))
        return cls(layout)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read a whitespace-separated grid of integers; lines starting with # are comments"""
        return cls(np.loadtxt(path, dtype=np.intp, ndmin=2))

    def to_surface(self, vector: ArrayLike) -> FloatArray:
        """Place one value per actuator, in vector order, onto the grid; other cells are 0

        :raises ValueError: if the vector does not hold one value per actuator
        """
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.num_actuators,):
            raise ValueError(f"Expected {self.num_actuators} values, got shape {values.shape}")
        surface = np.zeros(self.side * self.side)
        surface[self._cells] = values
        return surface.reshape(self.shape)

    def to_vector(self, surface: ArrayLike) -> FloatArray:
        """Read one value per actuator off the grid, in vector order

        :raises ValueError: if the surface is not the grid's shape
        """
        values = np.asarray(surface, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(f"Expected a {self.shape} surface, got shape {values.shape}")
        return values.reshape(-1)[self._cells]


# One field per config setting, so the record mirrors the TOML file
@dataclass(frozen=True, eq=False)
class DMConfig:  # pylint: disable=too-many-instance-attributes
    """Everything needed to drive one DM; read a bundle with load()

    Flats are read-only surfaces on the actuator map's grid. With last_command_path set,
    the controller saves every command there and loads it on connect.

    :raises ConfigError: if the soft limits, flats or default flat are invalid
    """

    name: str
    actuator_map: ActuatorMap
    soft_limits: tuple[float, float]
    serial_number: str | None = None
    flats: Mapping[str, FloatArray] = field(default_factory=dict)
    default_flat: str | None = None
    sdk_profiles_path: Path | None = None
    sdk_maps_path: Path | None = None
    logfile: str = DEFAULT_LOGFILE
    last_command_path: Path | None = None

    def __post_init__(self) -> None:
        """Check the limits and flats, and store the flats as read-only surfaces"""
        minimum, maximum = self.soft_limits
        if not HARDWARE_MIN <= minimum < maximum <= HARDWARE_MAX:
            raise ConfigError(
                f"Soft limits must lie in 0 to 1 with minimum < maximum, got {self.soft_limits}"
            )
        flats = {}
        for name, flat in self.flats.items():
            surface = np.array(flat, dtype=np.float64)
            if surface.shape != self.actuator_map.shape or not np.isfinite(surface).all():
                raise ConfigError(
                    f"Flat {name!r} must be a finite {self.actuator_map.shape} surface"
                )
            flats[name] = _read_only(surface)
        object.__setattr__(self, "flats", MappingProxyType(flats))
        if self.default_flat is not None and self.default_flat not in flats:
            raise ConfigError(f"Default flat {self.default_flat!r} is not in {sorted(flats)}")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a bundle: a TOML file with actuator_map.txt and its flats beside it

        File names in the TOML resolve against its folder. With a single flat and no
        default_flat, that flat is the default. The last command is saved beside the TOML
        as last_command.npy.

        :raises ConfigError: if the TOML, the actuator map or a flat is missing or invalid
        """
        source = Path(path).expanduser()
        folder = source.resolve().parent
        try:
            with source.open("rb") as file:
                document = tomllib.load(file)
            _check_schema(document)
            dm, limits, sdk, log = (
                document.get(name, {}) for name in ("dm", "limits", "sdk", "log")
            )
            flat_files: dict[str, str] = document.get("flats", {})
            only_flat = next(iter(flat_files)) if len(flat_files) == 1 else None
            actuator_map = ActuatorMap.from_file(folder / ACTUATOR_MAP_FILE)
            return cls(
                name=dm.get("name", source.stem),
                actuator_map=actuator_map,
                soft_limits=(float(limits["minimum"]), float(limits["maximum"])),
                serial_number=dm.get("serial_number"),
                flats={
                    name: _read_flat(folder / file, actuator_map)
                    for name, file in flat_files.items()
                },
                default_flat=dm.get("default_flat", only_flat),
                sdk_profiles_path=_resolve(folder, sdk.get("profiles_path")),
                sdk_maps_path=_resolve(folder, sdk.get("maps_path")),
                logfile=str(_resolve(folder, log.get("file")) or DEFAULT_LOGFILE),
                last_command_path=folder / LAST_COMMAND_FILE,
            )
        except (OSError, EOFError, ValueError) as error:
            raise ConfigError(f"{source}: {error}") from error


def _check_schema(document: Mapping[str, object]) -> None:
    """Reject unknown sections and keys, wrongly typed values and missing limits"""
    for section, table in document.items():
        if section not in _CONFIG_SCHEMA:
            raise ConfigError(f"unknown section [{section}]")
        if not isinstance(table, dict):
            raise ConfigError(f"[{section}] must be a table")
        for key, value in table.items():
            expected = str if section == "flats" else _CONFIG_SCHEMA[section].get(key)
            if expected is None:
                raise ConfigError(f"unknown key {key!r} in [{section}]")
            # TOML integers count as numbers, booleans do not
            number = expected is float and isinstance(value, int) and not isinstance(value, bool)
            if not (isinstance(value, expected) or number):
                raise ConfigError(f"[{section}] {key} must be a {expected.__name__}")
    limits = document.get("limits")
    if not isinstance(limits, dict) or not {"minimum", "maximum"} <= limits.keys():
        raise ConfigError("[limits] needs minimum and maximum")


def _resolve(folder: Path, name: str | None) -> Path | None:
    """Resolve a file name from the config against the config's folder"""
    return None if name is None else folder / Path(name).expanduser()


def _read_flat(path: Path, actuator_map: ActuatorMap) -> FloatArray:
    """Read a .npy surface, or a BMC .txt vector in actuator order"""
    try:
        match path.suffix:
            case ".npy":
                return np.asarray(np.load(path), dtype=np.float64)
            case ".txt":
                return actuator_map.to_surface(np.loadtxt(path, ndmin=1))
    except (OSError, EOFError, ValueError) as error:
        raise ConfigError(f"Flat {path}: {error}") from error
    raise ConfigError(f"Flat {path} must be .npy or .txt")


def actuators(actuator_map: ActuatorMap, indices: int | Sequence[int]) -> FloatArray:
    """Select actuators by command vector index (0-based)

    :raises TypeError: if an index is not an integer
    :raises ValueError: if an index is out of range
    """
    selected = np.atleast_1d(np.asarray(indices))
    if not np.issubdtype(selected.dtype, np.integer):
        raise TypeError(f"Actuator indices must be integers, got {indices!r}")
    if selected.min() < 0 or selected.max() >= actuator_map.num_actuators:
        raise ValueError(f"Actuator indices must be 0 to {actuator_map.num_actuators - 1}")
    vector = np.zeros(actuator_map.num_actuators)
    vector[selected] = 1.0
    return actuator_map.to_surface(vector)


def grid(actuator_map: ActuatorMap, row: int, column: int, spacing: int) -> FloatArray:
    """Select one cell per spacing x spacing block, at (row, column) within each block

    :raises ValueError: if row or column is negative or spacing is below 1
    """
    if min(row, column) < 0 or spacing < 1:
        raise ValueError(
            f"grid needs row, column >= 0 and spacing >= 1, got {row, column, spacing}"
        )
    pattern = np.zeros(actuator_map.shape)
    pattern[row::spacing, column::spacing] = 1.0
    return pattern


def rows(actuator_map: ActuatorMap, indices: int | Sequence[int]) -> FloatArray:
    """Select whole rows of the grid"""
    pattern = np.zeros(actuator_map.shape)
    pattern[np.asarray(indices), :] = 1.0
    return pattern


def columns(actuator_map: ActuatorMap, indices: int | Sequence[int]) -> FloatArray:
    """Select whole columns of the grid"""
    pattern = np.zeros(actuator_map.shape)
    pattern[:, np.asarray(indices)] = 1.0
    return pattern


def crosshair(actuator_map: ActuatorMap, width: int = 2) -> FloatArray:
    """Select a centered cross of bars width cells wide (an even width centers it)

    :raises ValueError: if width is below 1
    """
    if width < 1:
        raise ValueError(f"Crosshair width must be at least 1, got {width}")
    start = actuator_map.side // 2 - width // 2
    pattern = np.zeros(actuator_map.shape)
    pattern[:, start : start + width] = 1.0
    pattern[start : start + width, :] = 1.0
    return pattern


def letter_f(actuator_map: ActuatorMap) -> FloatArray:
    """Draw a capital F to check the DM orientation"""
    scale = actuator_map.side / _F_REFERENCE_SIDE
    pattern = np.zeros(actuator_map.shape)
    for row_start, row_stop, column_start, column_stop in _F_STROKES:
        pattern[
            round(row_start * scale) : round(row_stop * scale),
            round(column_start * scale) : round(column_stop * scale),
        ] = 1.0
    return pattern


def edge(actuator_map: ActuatorMap, start: int = 0, step: int = 1) -> FloatArray:
    """Select every step-th edge actuator, from actuator index start onward

    Use an odd step: with row-by-row numbering, edge indices alternate between the left
    and right sides, so an even step pokes mostly one side.
    """
    edges = actuator_map.edge_actuators
    vector = np.zeros(actuator_map.num_actuators)
    vector[edges[edges >= start][::step]] = 1.0
    return actuator_map.to_surface(vector)


def sine(
    actuator_map: ActuatorMap, frequency: float = 1.0, angle: float = 0.0, phase: float = 0.0
) -> FloatArray:
    """Return a sinusoid of frequency cycles across the grid, at angle and phase (radians)"""
    axis = np.linspace(-np.pi, np.pi, actuator_map.side)
    x, y = np.meshgrid(axis, axis)
    wave = np.sin(frequency * np.cos(angle) * x + frequency * np.sin(angle) * y + phase)
    return np.asarray(wave, dtype=np.float64)


def zernike_basis(actuator_map: ActuatorMap, max_noll: int) -> FloatArray:
    """Return Zernikes for Noll indices 1 to max_noll, so basis[noll - 1] is one term

    Every term except piston is scaled to a peak-to-valley of 1 over the actuators.
    Needs poppy (pip install "bmc_dm[zernike]").

    :raises ValueError: if max_noll is below 1
    """
    if max_noll < 1:
        raise ValueError(f"Noll indices start at 1, got {max_noll}")
    terms = max(_MIN_ZERNIKE_TERMS, 1 << (max_noll - 1).bit_length())
    basis = _poppy_zernikes(terms, actuator_map.side)[:max_noll].copy()
    basis[1:] /= np.ptp(basis[1:, actuator_map.mask], axis=1)[:, np.newaxis, np.newaxis]
    return basis


def zernike(actuator_map: ActuatorMap, noll: int) -> FloatArray:
    """Return one Zernike by Noll index (1 is piston), scaled as in zernike_basis"""
    return np.asarray(zernike_basis(actuator_map, noll)[noll - 1], dtype=np.float64)


def _random_zernike_sums(
    actuator_map: ActuatorMap, nolls: IndexArray, scales: FloatArray, steps: int
) -> FloatArray:
    """Return steps surfaces, each a sum of Zernikes with Gaussian amplitudes of std scales"""
    basis = zernike_basis(actuator_map, int(nolls.max()))[nolls - 1]
    coefficients = np.random.default_rng().normal(0.0, scales, (steps, nolls.size))
    return np.tensordot(coefficients, basis, axes=1)


@lru_cache(maxsize=4)
def _poppy_zernikes(terms: int, side: int) -> FloatArray:
    """Return poppy's unnormalized Zernikes on a side x side grid, cached and read-only"""
    # Imported here: poppy is a heavy, optional dependency that only Zernikes need
    import poppy  # pylint: disable=import-outside-toplevel,import-error

    # poppy zeroes outside its unit circle, so generate one cell wider on each side to
    # cover BMC's active region, then crop, as KPIC does
    basis = np.asarray(poppy.zernike.zernike_basis(terms, side + 2, outside=0.0), dtype=np.float64)
    return _read_only(basis[:, 1:-1, 1:-1])


class MirrorState(StrEnum):
    """What the mirror holds, inferred from the commands sent (BMC DMs have no readback)"""

    OFF = "off"  # Disconnected: the SDK zeroed the DACs and turned HV off
    ZEROED = "zeroed"  # Connected with nothing sent yet, so every actuator is at 0
    HOLDING = "holding"  # The last command is on the mirror


@dataclass(frozen=True)
class StatusReport:
    """Snapshot from BmcDmController.status_report(); command statistics are None when off"""

    mirror_state: MirrorState
    last_command_at: datetime | None  # UTC; None from connect() until the first command
    last_message: str
    minimum: float | None = None
    maximum: float | None = None
    at_limit: int | None = None  # Actuators at either soft limit
    max_neighbor_step: float | None = None  # Between grid neighbors, in command units

    def summary(self) -> str:
        """Return the report as one log line"""
        if self.minimum is None:
            return str(self.mirror_state)
        return (
            f"{self.mirror_state}: {self.minimum:.3f} to {self.maximum:.3f}, "
            f"{self.at_limit} at a limit, largest neighbor step {self.max_neighbor_step:.3f}"
        )


def _max_neighbor_step(actuator_map: ActuatorMap, vector: FloatArray) -> float:
    """Return the largest difference between actuators that are grid neighbors"""
    surface, mask = actuator_map.to_surface(vector), actuator_map.mask
    rows_step = np.abs(np.diff(surface, axis=0))[mask[1:] & mask[:-1]]
    columns_step = np.abs(np.diff(surface, axis=1))[mask[:, 1:] & mask[:, :-1]]
    return float(np.max(np.concatenate((rows_step, columns_step)), initial=0.0))


class BmcDmController(HardwareDeviceBase):  # type: ignore[misc]
    """Control a BMC deformable mirror through the BMC DM-SDK

    Build from a config bundle, then connect(). connect() and disconnect() send no
    commands; every command is checked (shape, finite values) and clipped to the soft
    limits before it reaches the SDK.

    Errors raise: DMError for SDK and connection failures, ValueError or TypeError for
    bad input, ConfigError for a bad bundle. Controller errors are also reported
    through report_error.
    """

    def __init__(self, config: str | Path | DMConfig, *, log: bool = True) -> None:
        """Load the config bundle; no hardware access until connect()

        :param config: path to the bundle's TOML file, or a DMConfig
        :param log: log to the console and the config's log file if True, otherwise print
        :raises ConfigError: if the bundle is invalid
        """
        self.config = config if isinstance(config, DMConfig) else DMConfig.load(config)
        if log:
            Path(self.config.logfile).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(log, self.config.logfile)
        # Reentrant so add() can read the surface and send the result as one step
        self.lock = threading.RLock()
        self._device: Any = None
        self._no_error = 0
        self._last_command: FloatArray | None = None
        self._last_command_at: datetime | None = None

    @property
    def actuator_map(self) -> ActuatorMap:
        """Return the config's actuator map, for building patterns"""
        return self.config.actuator_map

    @property
    def flat(self) -> FloatArray | None:
        """Return the config's default flat surface (not applied), or None without one"""
        name = self.config.default_flat
        return None if name is None else self.config.flats[name]

    def __enter__(self) -> Self:
        """Return the controller; connect() is still an explicit step"""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Disconnect if connected; the SDK zeroes the mirror as it closes (see disconnect)"""
        if self.connected:
            self.disconnect()

    def connect(self, serial_number: str | None = None) -> None:  # pylint: disable=arguments-differ
        """Open the DM through the SDK without sending any command

        Loads the command saved by the last session as the current command, falling back
        to the SDK's readback without one. Call restore() to send it again.

        :param serial_number: DM serial, defaults to the config's serial
        :raises DMError: if already connected, the SDK fails, or the actuator count differs
        :raises ValueError: if there is no serial number
        """
        serial = serial_number or self.config.serial_number
        if serial is None:
            self._reject("No serial number given and none in the config")
        try:
            # Imported on connect so this module loads without the SDK (tests, offline work)
            import bmc  # pylint: disable=import-outside-toplevel,import-error
        except ImportError:
            self._raise_dm_error(
                "BMC DM-SDK Python module (bmc) is not installed; add "
                "/opt/Boston Micromachines/lib/python3 to the environment"
            )
        with self.lock:
            if self._device is not None:
                self._raise_dm_error("DM is already connected")
            saved = self._read_saved_command()
            device = bmc.BmcDm()
            self._no_error = bmc.NO_ERR
            if (profiles := self.config.sdk_profiles_path) is not None:
                self._check(device, device.set_profiles_path(str(profiles)), f"set {profiles}")
            if (maps := self.config.sdk_maps_path) is not None:
                self._check(device, device.set_maps_path(str(maps)), f"set {maps}")
            self._check(device, device.open_dm(serial), f"open DM {serial}")
            if (count := device.num_actuators()) != self.actuator_map.num_actuators:
                device.close_dm()
                self._raise_dm_error(
                    f"DM {serial} has {count} actuators, the {self.config.name} actuator map "
                    f"has {self.actuator_map.num_actuators}"
                )
            self._device = device
            self._last_command = self._read_reply() if saved is None else saved
            self._last_command_at = None
            self._set_connected(True)
        self.report_info(
            f"Connected to {self.config.name} {serial}, {self.status_report().summary()}"
        )

    def disconnect(self) -> None:
        """Close the DM without sending a zero command

        The SDK's close_dm() zeroes and disables the DACs itself (BMC_PCIeClose and
        BMC_USBClose in the SDK source), so the mirror returns to zero.

        :raises DMError: if closing fails
        """
        with self.lock:
            device, self._device = self._device, None
            self._last_command = None
            if device is None:
                self.report_warning("DM is already disconnected")
                return
            self._set_connected(False)
            self.initialized = False
            self._check(device, device.close_dm(), "close the DM")
        self.report_info("Disconnected from DM")

    def initialize(self) -> bool:
        """Apply the default flat, or zero the DM without one

        :return: True once initialized
        """
        self.initialized = False
        if self.flat is None:
            self.zero()
        else:
            self.set_flat()
        self.initialized = True
        return True

    def set_vector(self, vector: ArrayLike) -> FloatArray:
        """Send one value per actuator, in command vector order

        :return: the surface that was applied, after clipping
        :raises ValueError: if the shape is wrong or a value is not finite
        :raises DMError: if the DM is not connected or the SDK fails
        """
        command = self._prepare(vector)
        self._send_command(command)
        return self.actuator_map.to_surface(command)

    def set_surface(self, surface: ArrayLike) -> FloatArray:
        """Send a surface; cells without an actuator are ignored

        :return: the surface that was applied, after clipping
        """
        surface_array = self._as_array(surface, self.actuator_map.shape, "Surface")
        return self.set_vector(self.actuator_map.to_vector(surface_array))

    def get_vector(self) -> FloatArray:
        """Return the last command, one value per actuator

        Right after connect() this is the command saved by the last session, which the
        mirror only holds again after restore().

        :raises DMError: if the DM is not connected
        """
        with self.lock:
            if self._last_command is None:
                self._raise_dm_error("DM is not connected")
            return self._last_command.copy()

    def get_surface(self) -> FloatArray:
        """Return the last command as a surface (see get_vector)"""
        return self.actuator_map.to_surface(self.get_vector())

    def restore(self) -> FloatArray:
        """Send the last command again, e.g. the one loaded on connect()

        :return: the surface that was applied, after clipping
        :raises DMError: if the DM is not connected
        """
        with self.lock:
            return self.set_vector(self.get_vector())

    def zero(self) -> FloatArray:
        """Set every actuator to 0"""
        return self.set_vector(np.zeros(self.actuator_map.num_actuators))

    def set_flat(self, name: str | None = None) -> FloatArray:
        """Apply a flat from the config, by default its default flat

        :return: the surface that was applied, after clipping
        :raises ValueError: if there is no such flat
        """
        flat_name = name or self.config.default_flat
        if flat_name is None or flat_name not in self.config.flats:
            self._reject(f"No flat {flat_name!r}; the config has {sorted(self.config.flats)}")
        return self.set_surface(self.config.flats[flat_name])

    def poke(
        self,
        pattern: Pattern | ArrayLike,
        amplitude: float,
        *,
        bias: ArrayLike | None = None,
        **pattern_args: Any,
    ) -> FloatArray:
        """Apply bias + amplitude * pattern

        pattern is a surface, or a pattern function that is called with the actuator map
        and pattern_args, e.g. poke(zernike, 0.2, noll=5). bias defaults to the default
        flat (zeros without one); bias=0 starts from zeros.

        :return: the surface that was applied, after clipping
        :raises ValueError: if pattern or bias is not a full surface
        """
        if callable(pattern):
            pattern = pattern(self.actuator_map, **pattern_args)
        elif pattern_args:
            self._reject(f"Pattern arguments {sorted(pattern_args)} need a pattern function")
        unit = self._as_array(pattern, self.actuator_map.shape, "Pattern")
        return self.set_surface(self._bias(bias) + amplitude * unit)

    def add(
        self, pattern: Pattern | ArrayLike, amplitude: float, **pattern_args: Any
    ) -> FloatArray:
        """Add amplitude * pattern to the last command (see poke and get_vector)"""
        with self.lock:
            return self.poke(pattern, amplitude, bias=self.get_surface(), **pattern_args)

    def flash_edge(
        self,
        amplitude: float = -0.2,
        pause_s: float = 0.3,
        iterations: int = 20,
        *,
        bias: ArrayLike | None = None,
    ) -> None:
        """Alternate pokes between two interleaved sets of edge actuators to show the edge

        Blocks for about 2 * pause_s * iterations seconds, then restores bias (as in
        poke), also after Ctrl-C or an error.
        """
        if pause_s < 0 or iterations < 0:
            self._reject("flash_edge needs pause_s >= 0 and iterations >= 0")
        surface = self._bias(bias)
        interleaved = [edge(self.actuator_map, start, _FLASH_EDGE_STEP) for start in (0, 1)]
        try:
            for iteration in range(iterations):
                for pattern in interleaved:
                    self.poke(pattern, amplitude, bias=surface)
                    time.sleep(pause_s)
                self.report_debug(f"Edge flash {iteration + 1} of {iterations}")
        finally:
            self.set_surface(surface)

    def simulate_turbulence(
        self,
        nolls: Sequence[int],
        amplitudes: float | Sequence[float] = 0.002,
        rate_hz: float = 5.0,
        duration_s: float = 10.0,
        *,
        bias: ArrayLike | None = None,
    ) -> FloatArray:
        """Apply random Zernike wavefront error on top of bias (as in poke)

        Each step draws every Zernike amplitude from a zero-mean Gaussian with standard
        deviation amplitudes (one value, or one per Noll index) and holds it for
        1 / rate_hz. Ctrl-C stops early. bias is restored when done.

        :return: the wavefronts applied, one surface per step, without the bias
        :raises ValueError: for no or below-1 Noll indices, negative or mismatched
            amplitudes, rate_hz <= 0, or rate_hz * duration_s below 1
        """
        noll_indices = np.atleast_1d(np.asarray(nolls, dtype=np.intp))
        scales = np.asarray(amplitudes, dtype=np.float64)
        steps = int(rate_hz * duration_s)
        # Check sizes before min(), which fails on empty arrays
        sizes_valid = noll_indices.size > 0 and scales.shape in ((), noll_indices.shape)
        if (
            rate_hz <= 0
            or steps < 1
            or not sizes_valid
            or noll_indices.min() < 1
            or scales.min() < 0
        ):
            self._reject(
                "simulate_turbulence needs Noll indices >= 1, one amplitude >= 0 (or one per "
                "Noll index), rate_hz > 0 and rate_hz * duration_s >= 1"
            )
        surface = self._bias(bias)
        wavefronts = _random_zernike_sums(self.actuator_map, noll_indices, scales, steps)
        start = time.monotonic()
        applied = 0
        try:
            for wavefront in wavefronts:
                self.set_surface(surface + wavefront)
                applied += 1
                # Hold every wavefront, the last included, until its period ends
                time.sleep(max(0.0, start + applied / rate_hz - time.monotonic()))
        except KeyboardInterrupt:
            self.report_info(f"Turbulence stopped after {applied} of {steps} steps")
        finally:
            self.set_surface(surface)
        return wavefronts[:applied]

    def status_report(self) -> StatusReport:
        """Summarize what the mirror was sent, without calling the SDK (safe from any thread)

        BMC mirrors cannot be read back. While ZEROED, the statistics describe the shape
        restore() would send.
        """
        with self.lock:
            command, sent_at = self._last_command, self._last_command_at
            if self._device is None:
                state = MirrorState.OFF
            elif sent_at is None:
                state = MirrorState.ZEROED
            else:
                state = MirrorState.HOLDING
            last_message = self.status_string
        if command is None:
            return StatusReport(
                mirror_state=state, last_command_at=sent_at, last_message=last_message
            )
        return StatusReport(
            mirror_state=state,
            last_command_at=sent_at,
            last_message=last_message,
            minimum=float(command.min()),
            maximum=float(command.max()),
            at_limit=int(np.count_nonzero(np.isin(command, self.config.soft_limits))),
            max_neighbor_step=_max_neighbor_step(self.actuator_map, command),
        )

    def _send_command(self, vector: FloatArray) -> bool:  # pylint: disable=arguments-differ
        """Send a prepared command vector to the SDK, then record and save it"""
        with self.lock:
            # The SDK segfaults on calls without an open DM, so never reach it unconnected
            if self._device is None:
                self._raise_dm_error("DM is not connected")
            # BMC's Python examples pass plain lists to send_data
            self._check(self._device, self._device.send_data(vector.tolist()), "send the command")
            self._last_command, self._last_command_at = vector, datetime.now(UTC)
            self._save_command(vector)
        return True

    def _save_command(self, vector: FloatArray) -> None:
        """Write the command to the config's last_command_path, warning if that fails"""
        path = self.config.last_command_path
        if path is None:
            return
        # Write a temporary file and rename it, so a crash never leaves a partial file
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("wb") as file:
                np.save(file, vector)
            temporary.replace(path)
        except OSError as error:
            # The mirror already has the command, so only the record of it is lost
            self.report_warning(f"Could not save the last command to {path}: {error}")

    def _read_saved_command(self) -> FloatArray | None:
        """Return the command saved at last_command_path, or None if there is no valid one"""
        path = self.config.last_command_path
        if path is None:
            return None
        try:
            saved = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        except FileNotFoundError:
            return None
        except (OSError, EOFError, ValueError) as error:
            self.report_warning(f"Ignoring saved command {path}: {error}")
            return None
        if saved.shape != (self.actuator_map.num_actuators,) or not np.isfinite(saved).all():
            self.report_warning(
                f"Ignoring saved command {path}: expected "
                f"{self.actuator_map.num_actuators} finite values, got shape {saved.shape}"
            )
            return None
        self.report_info(f"Loaded the last command from {path}; restore() sends it again")
        return saved

    def _read_reply(self) -> FloatArray:
        """Read the last command vector back from the SDK"""
        with self.lock:
            if self._device is None:
                self._raise_dm_error("DM is not connected")
            vector = np.asarray(self._device.get_actuator_data(), dtype=np.float64)
        if vector.shape != (self.actuator_map.num_actuators,):
            self._raise_dm_error(f"SDK returned {vector.size} actuator values")
        return vector

    def _prepare(self, vector: ArrayLike) -> FloatArray:
        """Check a command and clip it to the soft limits without changing the input"""
        command = self._as_array(vector, (self.actuator_map.num_actuators,), "Command")
        if not np.isfinite(command).all():
            self._reject("Command contains NaN or infinite values")
        minimum, maximum = self.config.soft_limits
        clipped = np.clip(command, minimum, maximum)
        if clipped_count := int(np.count_nonzero(clipped != command)):
            self.report_warning(
                f"Clipped {clipped_count} of {command.size} values to [{minimum}, {maximum}]"
            )
        return clipped

    def _bias(self, bias: ArrayLike | None) -> FloatArray:
        """Return bias as a surface: the default flat (or zeros) for None, a scalar fills it"""
        shape = self.actuator_map.shape
        if bias is None:
            return np.zeros(shape) if self.flat is None else self.flat
        return self._as_array(
            np.broadcast_to(bias, shape) if np.ndim(bias) == 0 else bias, shape, "Bias"
        )

    def _as_array(self, value: ArrayLike, shape: tuple[int, ...], label: str) -> FloatArray:
        """Convert value to a float array of shape, reporting non-numeric or misshapen input"""
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            self._reject(f"{label} is not numeric: {error}")
        if array.shape != shape:
            self._reject(f"{label} has shape {array.shape}, expected {shape}")
        return array

    def _check(self, device: Any, status: int, action: str) -> None:
        """Raise DMError with BMC's message if an SDK call did not return NO_ERR"""
        if status != self._no_error:
            self._raise_dm_error(f"Failed to {action}: {device.error_string(status)}")

    def _raise_dm_error(self, message: str) -> NoReturn:
        """Report an SDK or connection error, then raise it as DMError"""
        self.report_error(message)
        raise DMError(message)

    def _reject(self, message: str) -> NoReturn:
        """Report bad input, then raise it as ValueError"""
        self.report_error(message)
        raise ValueError(message)
