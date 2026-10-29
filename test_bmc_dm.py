"""Tests for the BMC DM driver against a fake BMC DM-SDK module"""

import sys
import threading
import types
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bmc_dm import (
    LAST_COMMAND_FILE,
    NO_ACTUATOR,
    ActuatorMap,
    BmcDmController,
    ConfigError,
    DMConfig,
    DMError,
    actuators,
    columns,
    crosshair,
    edge,
    grid,
    letter_f,
    rows,
    sine,
    zernike,
    zernike_basis,
)

NO_ERR = 0
ERR_NO_HW = 2
SIGHT_SERIAL = "30CW009#052"
SIGHT_CONFIG = Path(__file__).resolve().parent / "configs" / "sight" / "sight.toml"
SIGHT_MAP = ActuatorMap.circular(28)
SIGHT_FLATS = {"inst", "quick", "mean250", "mean400", "mean500", "mean600", "mean700"}
ZEROS = [0.0] * 648

# Edge actuators hardcoded in DM_Sight_20251008.py
SIGHT_EDGE_ACTUATORS = [
    *range(10),
    *(10, 11, 22, 23, 24, 25, 40, 41, 42, 61, 62, 83, 84, 107, 108, 131, 132, 157),
    *(158, 183, 184, 211, 212, 239, 240, 267, 268, 295, 296, 323, 324, 351, 352),
    *(379, 380, 407, 408, 435, 436, 463, 464, 489, 490, 515, 516, 539, 540, 563),
    *(564, 585, 586, 605, 606, 607, 622, 623, 624, 625, 636, 637),
    *range(638, 648),
]

BUNDLE_TOML = """
[dm]
name = "Test DM"
serial_number = "SERIAL"
default_flat = "surface"

[limits]
minimum = 0
maximum = 0.8

[flats]
surface = "surface.npy"
vector = "vector.txt"

[sdk]
profiles_path = "profiles"

[log]
file = "logs/dm"
"""


class FakeBmcDm:
    """Stand-in for bmc.BmcDm that records calls and returns preset error codes"""

    actuator_count = 648
    open_status = NO_ERR
    send_status = NO_ERR
    close_status = NO_ERR
    path_status = NO_ERR
    instances: ClassVar[list["FakeBmcDm"]] = []

    def __init__(self) -> None:
        """Start with no serial, paths or commands, and register the instance"""
        self.serial_number: str | None = None
        self.paths: dict[str, str] = {}
        self.sent: list[list[float]] = []
        self.close_count = 0
        self.sends_after_close = 0
        self.instances.append(self)

    def set_profiles_path(self, path: str) -> int:
        """Record the profiles path and return path_status"""
        self.paths["profiles"] = path
        return self.path_status

    def set_maps_path(self, path: str) -> int:
        """Record the maps path and return path_status"""
        self.paths["maps"] = path
        return self.path_status

    def open_dm(self, serial_number: str) -> int:
        """Record the serial and return open_status"""
        self.serial_number = serial_number
        return self.open_status

    def num_actuators(self) -> int:
        """Return actuator_count, which tests change to simulate the wrong DM"""
        return self.actuator_count

    def send_data(self, data: list[float]) -> int:
        """Record the command unless send_status is an error, counting sends after close"""
        if self.close_count:
            self.sends_after_close += 1
        if self.send_status == NO_ERR:
            self.sent.append(data)
        return self.send_status

    def get_actuator_data(self) -> list[float]:
        """Return the last command, or zeros before any, like the SDK"""
        return self.sent[-1] if self.sent else [0.0] * self.actuator_count

    def close_dm(self) -> int:
        """Count the close and return close_status"""
        self.close_count += 1
        return self.close_status

    def error_string(self, status: int) -> str:
        """Return a message naming the status code"""
        return f"fake error {status}"


@pytest.fixture(autouse=True)
def fixture_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake bmc module and reset its recorded instances"""
    monkeypatch.setitem(sys.modules, "bmc", types.SimpleNamespace(BmcDm=FakeBmcDm, NO_ERR=NO_ERR))
    monkeypatch.setattr(FakeBmcDm, "instances", [])


@pytest.fixture(name="flat")
def fixture_flat() -> np.ndarray:
    """Return a random flat that stays inside the soft limits under small pokes"""
    return SIGHT_MAP.to_surface(np.random.default_rng(0).uniform(0.2, 0.6, 648))


@pytest.fixture(name="config")
def fixture_config(flat: np.ndarray) -> DMConfig:
    """Return a Sight-like config whose default flat is flat"""
    return DMConfig(
        "Sight DM",
        SIGHT_MAP,
        (0.0, 0.8),
        serial_number=SIGHT_SERIAL,
        flats={"inst": flat},
        default_flat="inst",
    )


@pytest.fixture(name="dm")
def fixture_dm(config: DMConfig) -> BmcDmController:
    """Return a connected controller"""
    dm = BmcDmController(config, log=False)
    dm.connect()
    return dm


def _sdk() -> FakeBmcDm:
    """Return the single fake DM the test opened"""
    assert len(FakeBmcDm.instances) == 1
    return FakeBmcDm.instances[0]


def _bare_dm() -> BmcDmController:
    """Return a connected controller whose config has no flats"""
    dm = BmcDmController(DMConfig("DM", SIGHT_MAP, (0.0, 0.8)), log=False)
    dm.connect("serial")
    return dm


def _bundle(tmp_path: Path, toml: str = BUNDLE_TOML) -> Path:
    """Write a bundle (map, .npy and .txt flats, TOML) into tmp_path and return the TOML path"""
    np.savetxt(tmp_path / "actuator_map.txt", SIGHT_MAP.indices, fmt="%4d", header="test map")
    np.save(tmp_path / "surface.npy", np.where(SIGHT_MAP.mask, 0.5, 0.0))
    np.savetxt(tmp_path / "vector.txt", np.linspace(0, 0.6, 648))
    path = tmp_path / "dm.toml"
    path.write_text(toml)
    return path


def _sleep_raising_on_call(call: int) -> Callable[[float], None]:
    """Return a time.sleep stand-in that simulates Ctrl-C on its call-th call"""
    calls = 0

    def sleep(_seconds: float) -> None:
        """Return at once, except raise KeyboardInterrupt on the chosen call"""
        nonlocal calls
        calls += 1
        if calls == call:
            raise KeyboardInterrupt

    return sleep


# Actuator map


def test_sight_map_matches_sight_code() -> None:
    """Check the circular 28 map has 648 actuators and Sight's edge list"""
    assert SIGHT_MAP.shape == (28, 28) and SIGHT_MAP.num_actuators == 648
    assert SIGHT_MAP.edge_actuators.tolist() == SIGHT_EDGE_ACTUATORS


def test_kilo_dm_map() -> None:
    """Check the circular 34 map matches the 952-actuator kilo-DM"""
    kilo = ActuatorMap.circular(34)
    assert kilo.num_actuators == 952 and kilo.edge_actuators.size == 96


def test_map_vector_order_follows_indices() -> None:
    """Check values land on the cell holding their index, in any numbering"""
    custom = ActuatorMap([[2, NO_ACTUATOR], [0, 1]])
    surface = custom.to_surface([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(surface, [[30.0, 0.0], [10.0, 20.0]])
    np.testing.assert_array_equal(custom.to_vector(surface), [10.0, 20.0, 30.0])
    assert custom.edge_actuators.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "indices",
    [[[0, 1, 2]], [[0, 0], [1, -1]], [[0, 2], [-1, -1]], [[0, -2], [1, 2]], [[-1, -1], [-1, -1]]],
    ids=["not_square", "duplicate", "gap", "below_minus_one", "empty"],
)
def test_invalid_maps(indices: list[list[int]]) -> None:
    """Check maps that are not square or do not number 0 to N-1 once are rejected"""
    with pytest.raises(ValueError, match="Actuator map"):
        ActuatorMap(indices)


def test_map_arrays_are_read_only_copies() -> None:
    """Check the map copies its input and none of its arrays can be edited in place"""
    source = SIGHT_MAP.indices.copy()
    copied = ActuatorMap(source)
    source[0, 0] = 5
    assert copied.indices[0, 0] == NO_ACTUATOR
    for array in (copied.indices, copied.mask, copied.edge_actuators):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1


def test_map_shape_errors() -> None:
    """Check vectors and surfaces of the wrong size raise ValueError"""
    with pytest.raises(ValueError, match="648 values"):
        SIGHT_MAP.to_surface(np.zeros(647))
    with pytest.raises(ValueError, match="surface"):
        SIGHT_MAP.to_vector(np.zeros((34, 34)))


# Config


def test_load_bundle(tmp_path: Path) -> None:
    """Check a bundle loads its map and flats from beside the TOML and resolves paths"""
    config = DMConfig.load(_bundle(tmp_path))
    assert config.name == "Test DM" and config.serial_number == "SERIAL"
    assert config.soft_limits == (0.0, 0.8) and isinstance(config.soft_limits[0], float)
    np.testing.assert_array_equal(config.actuator_map.indices, SIGHT_MAP.indices)
    assert config.default_flat == "surface"
    assert config.sdk_profiles_path == tmp_path / "profiles" and config.sdk_maps_path is None
    assert config.logfile == str(tmp_path / "logs" / "dm")
    assert config.last_command_path == tmp_path / LAST_COMMAND_FILE
    np.testing.assert_array_equal(config.flats["surface"], np.where(SIGHT_MAP.mask, 0.5, 0.0))
    np.testing.assert_allclose(
        config.flats["vector"], SIGHT_MAP.to_surface(np.linspace(0, 0.6, 648))
    )
    assert not config.flats["surface"].flags.writeable


def test_minimal_bundle_defaults(tmp_path: Path) -> None:
    """Check optional keys default: name from the file, no serial, log in the working folder"""
    config = DMConfig.load(_bundle(tmp_path, "[limits]\nminimum = 0.0\nmaximum = 0.8\n"))
    assert config.name == "dm" and config.serial_number is None and not config.flats
    assert config.default_flat is None and config.logfile == "bmc_dm"


def test_single_flat_is_the_default(tmp_path: Path) -> None:
    """Check a bundle with one flat and no default_flat uses that flat"""
    toml = '[limits]\nminimum = 0.0\nmaximum = 0.8\n[flats]\nonly = "surface.npy"\n'
    assert DMConfig.load(_bundle(tmp_path, toml)).default_flat == "only"


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("[log]", "[extra]\nkey = 1\n[log]", r"unknown section \[extra\]"),
        ('name = "Test DM"', 'nmae = "Test DM"', "unknown key 'nmae'"),
        ("maximum = 0.8", "", "needs minimum and maximum"),
        ("maximum = 0.8", 'maximum = "0.8"', "maximum must be a float"),
        ("maximum = 0.8", "maximum = true", "maximum must be a float"),
        ('serial_number = "SERIAL"', "serial_number = 30", "serial_number must be a str"),
        ("maximum = 0.8", "maximum = 1.5", "Soft limits"),
        ('default_flat = "surface"', 'default_flat = "missing"', "Default flat"),
        ('surface = "surface.npy"', 'surface = "surface.fits"', ".npy or .txt"),
        ('surface = "surface.npy"', 'surface = "missing.npy"', "missing.npy"),
        ("[dm]", "[dm", "dm.toml"),
    ],
)
def test_config_errors(tmp_path: Path, old: str, new: str, match: str) -> None:
    """Check bad bundles raise ConfigError naming the problem"""
    assert old in BUNDLE_TOML
    with pytest.raises(ConfigError, match=match):
        DMConfig.load(_bundle(tmp_path, BUNDLE_TOML.replace(old, new)))


def test_config_file_errors(tmp_path: Path) -> None:
    """Check missing or invalid maps and malformed or wrongly sized flats raise ConfigError"""
    path = _bundle(tmp_path)
    (tmp_path / "surface.npy").write_bytes(b"not a numpy file")
    with pytest.raises(ConfigError, match="surface.npy"):
        DMConfig.load(path)
    np.save(tmp_path / "surface.npy", np.zeros((10, 10)))
    with pytest.raises(ConfigError, match="Flat 'surface'"):
        DMConfig.load(path)
    np.save(tmp_path / "surface.npy", np.zeros((28, 28)))
    np.savetxt(tmp_path / "vector.txt", np.zeros(100))
    with pytest.raises(ConfigError, match="vector.txt"):
        DMConfig.load(path)
    np.savetxt(tmp_path / "actuator_map.txt", np.zeros((28, 28)), fmt="%d")
    with pytest.raises(ConfigError, match="Actuator map"):
        DMConfig.load(path)
    (tmp_path / "actuator_map.txt").unlink()
    with pytest.raises(ConfigError, match="actuator_map.txt"):
        DMConfig.load(path)


def test_sight_bundle() -> None:
    """Check the shipped Sight bundle loads with its map, serial, limits and flats"""
    config = DMConfig.load(SIGHT_CONFIG)
    assert config.name == "Sight DM" and config.serial_number == SIGHT_SERIAL
    np.testing.assert_array_equal(config.actuator_map.indices, SIGHT_MAP.indices)
    assert config.soft_limits == (0.0, 0.8)
    assert set(config.flats) == SIGHT_FLATS and config.default_flat == "inst"
    for flat in config.flats.values():
        assert not flat[~SIGHT_MAP.mask].any() and 0.0 <= flat.min() and flat.max() <= 0.9


# Controller: connection


def test_constructor_loads_the_bundle_without_the_sdk() -> None:
    """Check a TOML path loads the bundle, exposes the flat and leaves the SDK alone"""
    dm = BmcDmController(SIGHT_CONFIG, log=False)
    assert not dm.is_connected() and not FakeBmcDm.instances
    assert dm.actuator_map.num_actuators == 648
    assert dm.flat is dm.config.flats["inst"]


def test_log_folder_created(tmp_path: Path) -> None:
    """Check a log file in a missing folder gets its folder created"""
    logfile = tmp_path / "logs" / "dm"
    BmcDmController(DMConfig("DM", SIGHT_MAP, (0.0, 0.8), logfile=str(logfile)))
    assert logfile.parent.is_dir()


def test_connect_sends_nothing(dm: BmcDmController) -> None:
    """Check connect opens the config's serial without any command or SDK path"""
    sdk = _sdk()
    assert dm.is_connected() and sdk.serial_number == SIGHT_SERIAL
    assert not sdk.sent and not sdk.paths


def test_connect_serial_override(config: DMConfig) -> None:
    """Check a serial passed to connect wins over the config's"""
    BmcDmController(config, log=False).connect("OTHER")
    assert _sdk().serial_number == "OTHER"


def test_connect_sets_sdk_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check configured SDK paths are set before opening, and failures raise DMError"""
    config = DMConfig(
        "DM", SIGHT_MAP, (0.0, 0.8), sdk_profiles_path=Path("/p"), sdk_maps_path=Path("/m")
    )
    BmcDmController(config, log=False).connect("serial")
    assert FakeBmcDm.instances[0].paths == {"profiles": "/p", "maps": "/m"}
    monkeypatch.setattr(FakeBmcDm, "path_status", ERR_NO_HW)
    with pytest.raises(DMError, match="set /p"):
        BmcDmController(config, log=False).connect("serial")


def test_connect_needs_a_serial() -> None:
    """Check a config without a serial requires serial_number"""
    dm = BmcDmController(DMConfig("DM", SIGHT_MAP, (0.0, 0.8)), log=False)
    with pytest.raises(ValueError, match="serial number"):
        dm.connect()


def test_connect_failures(config: DMConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check SDK open errors and actuator count mismatches raise DMError"""
    dm = BmcDmController(config, log=False)
    monkeypatch.setattr(FakeBmcDm, "open_status", ERR_NO_HW)
    with pytest.raises(DMError, match="fake error 2"):
        dm.connect()
    monkeypatch.setattr(FakeBmcDm, "open_status", NO_ERR)
    monkeypatch.setattr(FakeBmcDm, "actuator_count", 952)
    with pytest.raises(DMError, match="952 actuators"):
        dm.connect()
    assert FakeBmcDm.instances[-1].close_count == 1 and not dm.is_connected()
    status, message = dm.get_status()
    assert status == -1 and "952 actuators" in message


def test_connect_twice_raises(dm: BmcDmController) -> None:
    """Check connecting an already connected DM raises DMError"""
    with pytest.raises(DMError, match="already connected"):
        dm.connect()


def test_missing_sdk(config: DMConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check a missing bmc module raises DMError naming the SDK"""
    monkeypatch.setitem(sys.modules, "bmc", None)
    dm = BmcDmController(config, log=False)
    with pytest.raises(DMError, match="DM-SDK"):
        dm.connect()
    assert dm.get_status()[0] == -1


def test_commands_need_connection(config: DMConfig) -> None:
    """Check commands and reads raise DMError before connect"""
    dm = BmcDmController(config, log=False)
    with pytest.raises(DMError, match="not connected"):
        dm.zero()
    with pytest.raises(DMError, match="not connected"):
        dm.get_vector()
    with pytest.raises(DMError, match="not connected"):
        dm.restore()


def test_disconnect_keeps_shape(dm: BmcDmController) -> None:
    """Check disconnect sends no zero, closes once and a second call only warns"""
    dm.set_vector(np.full(648, 0.5))
    dm.disconnect()
    sdk = _sdk()
    assert sdk.sent[-1] == [0.5] * 648 and sdk.close_count == 1 and not dm.is_connected()
    dm.disconnect()
    assert sdk.close_count == 1


def test_close_failure_is_reported(dm: BmcDmController, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check a close error raises DMError and still leaves the DM disconnected"""
    monkeypatch.setattr(FakeBmcDm, "close_status", ERR_NO_HW)
    with pytest.raises(DMError, match="close the DM"):
        dm.disconnect()
    assert not dm.is_connected()


def test_context_manager_disconnects(config: DMConfig) -> None:
    """Check leaving a with block closes the DM without sending a zero"""
    with BmcDmController(config, log=False) as dm:
        dm.connect()
        dm.set_vector(np.full(648, 0.5))
    assert _sdk().sent[-1] == [0.5] * 648 and _sdk().close_count == 1


# Controller: last command


def test_commands_are_saved_beside_the_bundle(tmp_path: Path) -> None:
    """Check every command is saved, after clipping, as last_command.npy beside the TOML"""
    dm = BmcDmController(_bundle(tmp_path), log=False)
    dm.connect()
    command = np.full(648, 0.5)
    command[0] = 1.0
    dm.set_vector(command)
    saved = np.load(tmp_path / LAST_COMMAND_FILE)
    assert saved[0] == 0.8 and saved[1] == 0.5
    assert not list(tmp_path.glob(".*.tmp"))


def test_reconnect_loads_the_saved_command(tmp_path: Path) -> None:
    """Check a new session reads the saved command without sending, and restore sends it"""
    bundle = _bundle(tmp_path)
    first = BmcDmController(bundle, log=False)
    first.connect()
    first.set_vector(np.full(648, 0.5))
    first.disconnect()
    with pytest.raises(DMError, match="not connected"):
        first.get_vector()
    second = BmcDmController(bundle, log=False)
    second.connect()
    sdk = FakeBmcDm.instances[-1]
    assert not sdk.sent
    np.testing.assert_array_equal(second.get_vector(), 0.5)
    second.restore()
    assert sdk.sent == [[0.5] * 648]
    second.add(actuators, 0.1, indices=0)
    assert sdk.sent[-1][0] == pytest.approx(0.6) and sdk.sent[-1][1] == 0.5


@pytest.mark.parametrize(
    "write",
    [
        lambda path: np.save(path, np.zeros(10)),
        lambda path: np.save(path, np.full(648, np.nan)),
        lambda path: np.save(path, np.array(["a"] * 648)),
        lambda path: path.write_bytes(b"not a numpy file"),
    ],
    ids=["wrong_shape", "nan", "text", "not_npy"],
)
def test_bad_saved_command_falls_back_to_readback(
    tmp_path: Path, write: Callable[[Path], None], capsys: pytest.CaptureFixture[str]
) -> None:
    """Check an invalid saved command is ignored with a warning in favor of the SDK's readback"""
    bundle = _bundle(tmp_path)
    write(tmp_path / LAST_COMMAND_FILE)
    dm = BmcDmController(bundle, log=False)
    dm.connect()
    assert "Ignoring saved command" in capsys.readouterr().out
    np.testing.assert_array_equal(dm.get_vector(), ZEROS)


def test_save_failure_only_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check a command still goes out and is tracked when it cannot be saved"""
    bundle = _bundle(tmp_path)
    (tmp_path / LAST_COMMAND_FILE).mkdir()
    dm = BmcDmController(bundle, log=False)
    dm.connect()
    dm.set_vector(np.full(648, 0.5))
    assert "Could not save the last command" in capsys.readouterr().out
    assert _sdk().sent[-1] == [0.5] * 648
    np.testing.assert_array_equal(dm.get_vector(), 0.5)


# Controller: commands


def test_set_vector_clips_and_reads_back(
    dm: BmcDmController, capsys: pytest.CaptureFixture[str]
) -> None:
    """Check commands are clipped to the soft limits, sent as lists and read back"""
    command = np.full(648, 1.0)
    command[0] = -0.5
    surface = dm.set_vector(command)
    sent = _sdk().sent[-1]
    assert isinstance(sent, list) and sent[0] == 0.0 and sent[1] == 0.8
    assert command[0] == -0.5 and command[1] == 1.0
    np.testing.assert_array_equal(dm.get_surface(), surface)
    assert "Clipped 648 of 648" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command, match",
    [
        (np.zeros(647), "shape"),
        (np.zeros((28, 28)), "shape"),
        (np.full(648, np.nan), "NaN"),
        (["a"] * 648, "not numeric"),
    ],
    ids=["short", "surface", "nan", "text"],
)
def test_bad_commands_not_sent(dm: BmcDmController, command: object, match: str) -> None:
    """Check bad commands raise ValueError, are reported and never reach the SDK"""
    with pytest.raises(ValueError, match=match):
        dm.set_vector(command)  # type: ignore[arg-type]
    assert not _sdk().sent and dm.get_status()[0] == -1


def test_send_failure_raises(dm: BmcDmController, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check an SDK send error raises DMError with BMC's message"""
    monkeypatch.setattr(FakeBmcDm, "send_status", ERR_NO_HW)
    with pytest.raises(DMError, match="fake error 2"):
        dm.set_vector(np.full(648, 0.5))


def test_set_surface_ignores_cells_without_actuators(dm: BmcDmController) -> None:
    """Check values outside the actuator map are dropped"""
    surface = dm.set_surface(np.full((28, 28), 0.4))
    assert not surface[~SIGHT_MAP.mask].any() and np.all(dm.get_vector() == 0.4)


def test_set_flat(dm: BmcDmController, flat: np.ndarray) -> None:
    """Check set_flat applies the default or a named flat and rejects unknown names"""
    np.testing.assert_allclose(dm.set_flat(), flat)
    np.testing.assert_allclose(dm.set_flat("inst"), flat)
    with pytest.raises(ValueError, match="No flat 'missing'"):
        dm.set_flat("missing")
    with pytest.raises(ValueError, match="No flat None"):
        _bare_dm().set_flat()


def test_sight_flat_is_clipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Check the 0.9 peak of the Sight mean700 flat is clipped to 0.8"""
    # Save the command in tmp_path, not in the repo's Sight bundle
    config = replace(DMConfig.load(SIGHT_CONFIG), last_command_path=tmp_path / LAST_COMMAND_FILE)
    with BmcDmController(config, log=False) as dm:
        dm.connect()
        assert dm.set_flat("mean700").max() == 0.8
    assert "Clipped" in capsys.readouterr().out


def test_initialize_applies_default_flat_or_zero(config: DMConfig, flat: np.ndarray) -> None:
    """Check initialize applies the default flat, or zeros without one"""
    with BmcDmController(config, log=False) as dm:
        dm.connect()
        assert dm.initialize() and dm.is_initialized()
        np.testing.assert_allclose(dm.get_surface(), flat)
    bare = _bare_dm()
    bare.set_vector(np.full(648, 0.5))
    bare.initialize()
    assert not bare.get_vector().any()


def test_poke_biases(dm: BmcDmController, flat: np.ndarray) -> None:
    """Check poke offsets the default flat, a given bias, or zeros with bias=0"""
    flat_vector = SIGHT_MAP.to_vector(flat)
    pattern = actuators(SIGHT_MAP, [390, 393, 395])
    dm.poke(pattern, -0.1)
    np.testing.assert_allclose(dm.get_vector()[[390, 393, 395]], flat_vector[[390, 393, 395]] - 0.1)
    assert dm.get_vector()[391] == pytest.approx(flat_vector[391])
    dm.poke(pattern, 0.1, bias=0)
    assert np.flatnonzero(dm.get_vector()).tolist() == [390, 393, 395]
    dm.poke(pattern, 0.1, bias=np.full((28, 28), 0.3))
    assert dm.get_vector()[0] == pytest.approx(0.3)
    _bare_dm().poke(pattern, 0.1)
    assert np.flatnonzero(FakeBmcDm.instances[-1].sent[-1]).tolist() == [390, 393, 395]


def test_poke_with_pattern_functions(dm: BmcDmController, flat: np.ndarray) -> None:
    """Check pattern functions get the actuator map and keyword arguments"""
    surface = dm.poke(actuators, 0.1, indices=[0, 5])
    np.testing.assert_allclose(surface - flat, actuators(SIGHT_MAP, [0, 5]) * 0.1, atol=1e-12)
    dm.poke(crosshair, -0.1)
    np.testing.assert_allclose(dm.get_surface(), flat - 0.1 * crosshair(SIGHT_MAP) * SIGHT_MAP.mask)
    with pytest.raises(ValueError, match="need a pattern function"):
        dm.poke(crosshair(SIGHT_MAP), 0.1, width=4)


def test_add_builds_on_last_command(dm: BmcDmController) -> None:
    """Check add offsets whatever was sent last, with pattern functions too"""
    dm.set_vector(np.full(648, 0.3))
    dm.add(actuators, 0.05, indices=0)
    assert dm.get_vector()[0] == pytest.approx(0.35) and dm.get_vector()[1] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "pattern, bias",
    [(np.full(28, 1.0), None), (1.0, None), (crosshair(SIGHT_MAP), np.full(28, 0.5))],
    ids=["row_pattern", "scalar_pattern", "row_bias"],
)
def test_poke_rejects_partial_surfaces(dm: BmcDmController, pattern: object, bias: object) -> None:
    """Check patterns and biases other than scalars are never broadcast across the DM"""
    with pytest.raises(ValueError, match="shape"):
        dm.poke(pattern, 0.1, bias=bias)  # type: ignore[arg-type]
    assert not _sdk().sent


def test_flash_edge_restores_bias(
    dm: BmcDmController, flat: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check flash_edge alternates edge sets on the flat and restores it, also on Ctrl-C"""
    monkeypatch.setattr("bmc_dm.time.sleep", lambda _seconds: None)
    dm.flash_edge(-0.1, iterations=2)
    sent = _sdk().sent
    assert len(sent) == 2 * 2 + 1
    flat_vector = SIGHT_MAP.to_vector(flat)
    for start, command in ((0, sent[0]), (1, sent[1])):
        poked = np.flatnonzero(flat_vector - np.asarray(command) > 1e-9)
        assert poked.tolist() == [index for index in SIGHT_EDGE_ACTUATORS if index >= start][::3]
    np.testing.assert_allclose(dm.get_surface(), flat)

    monkeypatch.setattr("bmc_dm.time.sleep", _sleep_raising_on_call(3))
    with pytest.raises(KeyboardInterrupt):
        dm.flash_edge(-0.1)
    np.testing.assert_allclose(dm.get_surface(), flat)
    with pytest.raises(ValueError):
        dm.flash_edge(pause_s=-1)


def test_simulate_turbulence(
    dm: BmcDmController, flat: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check each step applies the flat plus a Zernike sum, is held, and the flat returns"""
    pytest.importorskip("poppy")
    sleeps: list[float] = []
    monkeypatch.setattr("bmc_dm.time.sleep", sleeps.append)
    wavefronts = dm.simulate_turbulence([4, 5, 6], amplitudes=0.01, rate_hz=10.0, duration_s=0.5)
    assert wavefronts.shape == (5, 28, 28)
    sent = _sdk().sent
    assert len(sent) == 5 + 1 and len(sleeps) == 5
    np.testing.assert_allclose(sent[0], np.clip(SIGHT_MAP.to_vector(flat + wavefronts[0]), 0, 0.8))
    np.testing.assert_allclose(sent[-1], SIGHT_MAP.to_vector(flat))
    basis = zernike_basis(SIGHT_MAP, 6)[[3, 4, 5]]
    coefficients = np.linalg.lstsq(basis.reshape(3, -1).T, wavefronts[0].ravel(), rcond=None)[0]
    np.testing.assert_allclose(np.tensordot(coefficients, basis, axes=1), wavefronts[0], atol=1e-12)

    monkeypatch.setattr("bmc_dm.time.sleep", _sleep_raising_on_call(4))
    stopped = dm.simulate_turbulence([4], rate_hz=10.0, duration_s=1.0, bias=0)
    assert stopped.shape == (4, 28, 28)
    assert _sdk().sent[-1] == ZEROS


@pytest.mark.parametrize(
    "nolls, amplitudes, rate_hz, duration_s",
    [
        ([0, 4], 0.002, 5.0, 10.0),
        ([], 0.002, 5.0, 10.0),
        ([4], 0.002, 0.05, 10.0),
        ([4], 0.002, -5.0, -10.0),
        ([4], -0.1, 5.0, 10.0),
        ([4], [], 5.0, 10.0),
        ([4, 5], [0.1, 0.1, 0.1], 5.0, 10.0),
    ],
    ids=["noll", "no_nolls", "steps", "negative_rate", "negative", "no_amplitudes", "mismatched"],
)
def test_simulate_turbulence_rejects(
    dm: BmcDmController,
    nolls: list[int],
    amplitudes: object,
    rate_hz: float,
    duration_s: float,
) -> None:
    """Check bad Noll indices, amplitudes, rates and runs too short for one step are rejected"""
    with pytest.raises(ValueError, match="simulate_turbulence needs"):
        dm.simulate_turbulence(nolls, amplitudes, rate_hz, duration_s)  # type: ignore[arg-type]
    assert not _sdk().sent and dm.get_status()[0] == -1


def test_disconnect_while_other_threads_send(dm: BmcDmController) -> None:
    """Check concurrent senders only see DMError and nothing reaches the SDK after close"""
    errors: list[BaseException] = []
    started = threading.Barrier(5)

    def keep_sending() -> None:
        """Send until the DM disconnects, recording any error other than DMError"""
        started.wait()
        try:
            while True:
                dm.add(actuators, 0.0, indices=0)
        except DMError:
            return
        # Record anything unexpected: an exception raised in a thread is otherwise lost
        except BaseException as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            errors.append(error)

    senders = [threading.Thread(target=keep_sending) for _ in range(4)]
    for sender in senders:
        sender.start()
    started.wait()
    dm.disconnect()
    for sender in senders:
        sender.join(timeout=5)

    assert not errors
    assert _sdk().close_count == 1 and _sdk().sends_after_close == 0


# Patterns


def test_actuators() -> None:
    """Check actuators selects 0-based vector indices and rejects bad ones"""
    vector = SIGHT_MAP.to_vector(actuators(SIGHT_MAP, [0, 647]))
    assert np.flatnonzero(vector).tolist() == [0, 647]
    with pytest.raises(ValueError):
        actuators(SIGHT_MAP, [648])
    with pytest.raises(ValueError):
        actuators(SIGHT_MAP, [-1])
    with pytest.raises(TypeError):
        actuators(SIGHT_MAP, [1.5])  # type: ignore[list-item]


def test_grid() -> None:
    """Check grid selects one cell per block at the given offset"""
    pattern = grid(SIGHT_MAP, 1, 2, 4)
    selected_rows, selected_columns = np.nonzero(pattern)
    assert set(selected_rows) == set(range(1, 28, 4))
    assert set(selected_columns) == set(range(2, 28, 4))
    assert pattern.sum() == 7 * 7
    with pytest.raises(ValueError):
        grid(SIGHT_MAP, 0, 0, 0)


def test_rows_and_columns() -> None:
    """Check rows and columns select whole lines of the grid"""
    assert np.nonzero(rows(SIGHT_MAP, [0, 5]).any(axis=1))[0].tolist() == [0, 5]
    assert np.nonzero(columns(SIGHT_MAP, 3).any(axis=0))[0].tolist() == [3]
    assert rows(SIGHT_MAP, [0, 5]).sum() == 2 * 28


def test_crosshair_is_centered() -> None:
    """Check the default crosshair covers the two middle rows and columns"""
    pattern = crosshair(SIGHT_MAP)
    assert np.nonzero(pattern.all(axis=1))[0].tolist() == [13, 14]
    assert np.nonzero(pattern.all(axis=0))[0].tolist() == [13, 14]
    assert pattern.sum() == 2 * 28 * 2 - 4
    with pytest.raises(ValueError):
        crosshair(SIGHT_MAP, 0)


def test_letter_f_matches_kpic_on_34_grid() -> None:
    """Check the scaled F reproduces KPIC's F on the 34-wide kilo-DM"""
    kilo = ActuatorMap.circular(34)
    expected = np.zeros(kilo.shape)
    expected[5:29, 10:13] = 1.0
    expected[5:8, 13:25] = 1.0
    expected[14:17, 13:25] = 1.0
    np.testing.assert_array_equal(letter_f(kilo), expected)


def test_letter_f_is_centered_on_sight() -> None:
    """Check the F is centered on the 28 grid and fully on actuators"""
    pattern = letter_f(SIGHT_MAP)
    on_rows, on_columns = np.nonzero(pattern)
    center = (28 - 1) / 2
    assert abs((on_rows.min() + on_rows.max()) / 2 - center) <= 0.5
    assert abs((on_columns.min() + on_columns.max()) / 2 - center) <= 0.5
    assert not pattern[~SIGHT_MAP.mask].any()


def test_edge_counts_from_actuator_index() -> None:
    """Check edge starts at an actuator index, as Sight does, then takes every step-th"""
    every_third = np.flatnonzero(SIGHT_MAP.to_vector(edge(SIGHT_MAP, 1, 3)))
    assert every_third.tolist() == SIGHT_EDGE_ACTUATORS[1::3]
    from_13 = [index for index in SIGHT_EDGE_ACTUATORS if index >= 13]
    assert np.flatnonzero(SIGHT_MAP.to_vector(edge(SIGHT_MAP, 13))).tolist() == from_13


def test_sine() -> None:
    """Check sine runs along the columns at angle 0 and along the rows at a quarter turn"""
    along_columns = sine(SIGHT_MAP, 2.0)
    np.testing.assert_array_equal(along_columns[0], along_columns[27])
    np.testing.assert_allclose(sine(SIGHT_MAP, 2.0, np.pi / 2), along_columns.T, atol=1e-12)
    np.testing.assert_allclose(sine(SIGHT_MAP, 1.0)[0, [0, 27]], [0.0, 0.0], atol=1e-12)


def test_zernikes() -> None:
    """Check Zernikes are peak-to-valley normalized and shared across basis sizes"""
    pytest.importorskip("poppy")
    basis = zernike_basis(SIGHT_MAP, 22)
    assert basis.shape == (22, 28, 28)
    np.testing.assert_allclose(np.ptp(basis[1:, SIGHT_MAP.mask], axis=1), 1.0)
    np.testing.assert_allclose(basis[0][SIGHT_MAP.mask], 1.0)
    np.testing.assert_array_equal(zernike(SIGHT_MAP, 4), zernike_basis(SIGHT_MAP, 40)[3])
    with pytest.raises(ValueError):
        zernike(SIGHT_MAP, 0)
