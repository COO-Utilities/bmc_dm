# bmc_dm

Python driver for Boston Micromachines (BMC) deformable mirrors (DMs), built on BMC's
DM-SDK and `hardware_device_base`. Every mirror is described by a config bundle: a TOML
file with the actuator map, the flats and the last command beside it. This README covers
everything needed to install, configure and run a mirror without reading the driver.

Start with [the three mirror files](#the-three-mirror-files): they are all a new mirror
needs besides the software.

Contents:

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Concepts](#3-concepts)
4. [Config bundle](#4-config-bundle)
5. [Controller reference](#5-controller-reference)
6. [Pattern functions](#6-pattern-functions)
7. [ActuatorMap](#7-actuatormap)
8. [Errors and logging](#8-errors-and-logging)
9. [How the SDK behaves](#9-how-the-sdk-behaves)
10. [The Sight bundle](#10-the-sight-bundle)
11. [Trying it without a mirror](#11-trying-it-without-a-mirror)
12. [Troubleshooting](#12-troubleshooting)
13. [Development](#13-development)

## The three mirror files

Every mirror needs three files. Everything else is software (section 1) or settings in a
short TOML file that ties the three together (section 4).

| | File | What it says | Where it comes from | Where it goes | Needed? |
| --- | --- | --- | --- | --- | --- |
| 1 | `actuator_map.txt` | Which actuator sits at which cell of the mirror's square grid | Written once per mirror layout; BMC's circular kilo-DMs can be generated (below) | Bundle folder, beside the TOML | Yes |
| 2 | Flat, e.g. `flat.npy` | The command that makes this mirror optically flat | Measured on your optical setup, e.g. with a wavefront sensor | Bundle folder, named in `[flats]` | For a flat mirror, yes; the driver also runs without one (pokes then start from zeros) |
| 3 | Profile, `<serial>.dm`, e.g. `30CW009#052.dm` | The mirror and its driver electronics, for the SDK | BMC, with the mirror; one per serial number | SDK profiles folder, `/opt/Boston Micromachines/Profiles/` | Yes: `connect()` fails without it |

### File 1: actuator map

A plain text grid of integers: each cell holds the number of the actuator at that
position (counted from 0, in BMC's actuator order), `-1` where there is no actuator. It
lets the driver draw patterns on the grid and send them in the order the SDK expects.
Full format in [section 4.3](#43-actuator-map-actuator_maptxt). For BMC's circular
kilo-DMs (648 actuators on 28 x 28, 952 on 34 x 34), generate it:

```python
import numpy as np
from bmc_dm import ActuatorMap

np.savetxt("actuator_map.txt", ActuatorMap.circular(28).indices, fmt="%4d")
```

Do not confuse it with the SDK's own `.map` files in `/opt/Boston Micromachines/Map`.
Those map driver channels to actuators (the wiring), are installed with the SDK, and are
used inside the SDK only.

### File 2: flat

A `side x side` numpy surface (`.npy`), or a vector of one value per actuator in BMC's
text format (`.txt`), with values from 0 to 1. A flat belongs to one mirror on one
optical setup, so measure it there. A bundle can hold several, each with a name;
the one named by `default_flat` (or the only one) is applied by `set_flat()` and is the
base for every poke. Format in [section 4.4](#44-flats).

### File 3: profile (`.dm`)

BMC's description of the mirror: interface type (USB or PCIe), actuator count, voltage
limits, the wiring map to use and any inactive actuators. The SDK reads it when
`connect()` opens the mirror, finding it by serial number, so keep BMC's file name. The
file is encoded (each byte XOR-ed with `0x83`); never edit it, but you can read it:

```python
from pathlib import Path

raw = Path("/opt/Boston Micromachines/Profiles/30CW009#052.dm").read_bytes()
print(bytes(b if b == 0x0A else b ^ 0x83 for b in raw).decode())
```

`type` is the interface (0 USB, 1 PCIe) and `size` the actuator count, which must match
the actuator map. If BMC also supplied a `.map` file, copy it to
`/opt/Boston Micromachines/Map/`. A calibration table (`.mat`) is not used by this driver.

### Putting them together

```
my_dm/
  my_dm.toml
  actuator_map.txt
  flat.npy
```

```toml
[dm]
serial_number = "30CW009#052"   # picks <serial>.dm from the SDK's profiles folder

[limits]
minimum = 0.0
maximum = 0.8

[flats]
flat = "flat.npy"               # the only flat, so it is the default
```

```python
from bmc_dm import BmcDmController

dm = BmcDmController("my_dm/my_dm.toml")
dm.connect()
dm.set_flat()
```

## 1. Install

You need:

- the BMC DM-SDK, with its Python module `bmc`
- your mirror's profile file (`<serial>.dm`), supplied by BMC with the mirror
- Python 3.12 or newer, the same Python the SDK's module was built for
- this driver, which pulls in numpy, `hardware_device_base` and optionally poppy

### 1.1 BMC DM-SDK (suggested: the Ubuntu 26.04 build)

Any DM-SDK 4.x install works as long as `import bmc` works in the driver's Python. On
Ubuntu 26.04 the suggested source is `dm-sdk-4.1.3-ubuntu26.tar.gz`: BMC's DM-SDK 4.1.3
source, patched to build with Ubuntu 26.04's compilers and libraries, Python 3.14 and
kernel 7.x. It replaces BMC's `.run` installer, which targets Ubuntu 18.04. The
`README.md` inside the tarball lists every change made to BMC's source.

1. Install the build prerequisites:

   ```bash
   sudo apt install build-essential qtbase5-dev qt5-qmake liblog4cxx-dev \
     libboost-dev libncurses-dev libpython3-dev zlib1g-dev libusb-1.0-0-dev \
     swig libopencv-dev libtbb12 libatlas3-base
   sudo apt install linux-headers-$(uname -r)   # PCIe interface card only
   ```

2. Extract the tarball into a folder **whose path has no spaces**. The SDK's build files
   do not quote paths, so a space anywhere in the path breaks the Python (SWIG) step and
   the install stops:

   ```bash
   mkdir -p ~/src
   tar xzf dm-sdk-4.1.3-ubuntu26.tar.gz -C ~/src
   cd ~/src/dm-sdk-4.1.3-ubuntu26
   ```

3. Build (no sudo needed). Add `--pcie` only for the PCIe interface card:

   ```bash
   ./build.sh            # USB drivers (Multi, Mini)
   ./build.sh --pcie     # also the PCIe kernel module
   ```

4. Install into `/opt/Boston Micromachines`:

   ```bash
   sudo ./install.sh          # USB
   sudo ./install.sh --pcie   # PCIe card
   ```

   This installs the libraries, programs, headers, examples, the Python module
   (`/opt/Boston Micromachines/lib/python3/bmc`), and BMC's generic profiles, maps,
   shapes and calibration tables. It registers the library folder with `ldconfig` and
   installs udev rules, so using the mirror needs no root access. Existing profiles are
   never overwritten.

5. Add your mirror's files, keeping BMC's file names:

   ```bash
   sudo cp 30CW009#052.dm "/opt/Boston Micromachines/Profiles/"
   sudo cp <calibration>.mat "/opt/Boston Micromachines/Calibration/"   # only if BMC supplied one
   ```

   For a USB driver, unplug and replug it once so the new udev permissions apply.

PCIe notes: the kernel module is built for the running kernel, so rerun
`./build.sh --pcie && sudo ./install.sh --pcie` after every kernel update. If `modprobe`
fails with "Key was rejected by service", Secure Boot is blocking the unsigned module:
disable Secure Boot or sign the module with a MOK key.

### 1.2 The driver

```bash
git clone git@github.com:COO-Utilities/bmc_dm.git
cd bmc_dm
python3 -m venv .venv            # use the system python3 the SDK was built with
source .venv/bin/activate
pip install -e ".[zernike]"      # zernike (poppy) is needed for Zernikes and turbulence
pip install -e ".[dev,zernike]"  # instead, to also run the tests and linters
```

`hardware_device_base` installs from GitHub, so the machine needs access to
`github.com/COO-Utilities`.

Make the SDK's Python module visible inside the venv (once per venv):

```bash
echo "/opt/Boston Micromachines/lib/python3" \
  > "$(python -c 'import site; print(site.getsitepackages()[0])')/bmc_sdk.pth"
```

### 1.3 Check the install

```bash
python -c "import bmc; print(bmc.BmcDm().version_string())"   # prints 4.1.3
python -c "from bmc_dm import BmcDmController; BmcDmController('configs/sight/sight.toml', log=False)"
```

The first line loads the SDK without touching the mirror. The second loads and checks the
Sight config bundle, also without hardware.

## 2. Quick start

```python
from bmc_dm import BmcDmController, actuators, letter_f, zernike

with BmcDmController("configs/sight/sight.toml") as dm:
    dm.connect()  # opens the mirror, sends nothing
    print(dm.get_surface())  # the previous session's last command
    dm.restore()  # send it again (the SDK zeroed it on close)

    dm.set_flat()  # apply the default flat
    dm.poke(letter_f, -0.1)  # flat - 0.1 on an F, to check orientation
    dm.poke(zernike, 0.2, noll=5)  # flat + 0.2 x astigmatism
    dm.add(zernike, 0.05, noll=4)  # plus 0.05 x focus, on top of the last command
    dm.poke(actuators, -0.1, indices=[390, 393, 395])
    dm.flash_edge()  # blink the pupil edge, then restore the flat
# leaving the block disconnects; the SDK zeroes the mirror as it closes
```

## 3. Concepts

**Commands are normalized.** Every actuator takes a value from 0 to 1, BMC's normalized
range; the SDK converts it to a voltage using the mirror's profile. Every command is
clipped to the bundle's soft limits (Sight: 0.0 to 0.8) before it is sent, and a warning
gives the number of values clipped.

**Vectors and surfaces.** A command is either:

- a *vector*: one value per actuator, `num_actuators` long, in BMC's actuator order (the
  order the SDK expects); index `i` is actuator `i`, counted from 0;
- a *surface*: a square `side x side` grid as the mirror is laid out. Cells without an
  actuator are ignored when a surface is sent and are 0 in surfaces the driver returns.

**The actuator map** links the two. It says which actuator sits at each grid cell, so
patterns can be drawn on the grid and sent in BMC's order. See
[Actuator map](#43-actuator-map-actuator_maptxt).

**Flats** are surfaces stored in the bundle that make the mirror optically flat. One of
them is the *default flat*: `set_flat()` applies it, and pokes are made on top of it unless
you pass another `bias`.

**The last command** is saved in the bundle as `last_command.npy` after every command, and
loaded on the next `connect()`. It is the driver's memory of the mirror's shape across
sessions, because the SDK can neither read the mirror back nor keep its shape after
closing (see [How the SDK behaves](#9-how-the-sdk-behaves)).

## 4. Config bundle

### 4.1 Layout

Everything for one mirror sits in one folder:

```
configs/sight/
  sight.toml           settings (any name)
  actuator_map.txt     required, always this name
  inst.npy  mean400.npy  ...   flats, named in the TOML
  last_command.npy     written by the driver (git-ignored)
```

Load it by passing the TOML path to `BmcDmController` (or to `DMConfig.load`). The whole
bundle is checked when it loads, before any hardware is touched; any problem raises
`ConfigError` naming the file.

### 4.2 TOML reference

```toml
[dm]
name = "Sight DM"
serial_number = "30CW009#052"
default_flat = "inst"

[limits]
minimum = 0.0
maximum = 0.8

[flats]
inst = "inst.npy"
quick = "quick.npy"

[sdk]
profiles_path = "/opt/Boston Micromachines/Profiles"
maps_path = "/opt/Boston Micromachines/Map"

[log]
file = "logs/bmc_dm.log"
```

| Key | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `[dm] name` | string | no | TOML file name without `.toml` | Name used in log messages |
| `[dm] serial_number` | string | no | none | BMC serial (11 characters); `connect("...")` overrides it, and one of the two is required |
| `[dm] default_flat` | string | no | the only flat, if `[flats]` has exactly one | Flat used by `set_flat()`, `initialize()` and as the default bias; must be a name in `[flats]` |
| `[limits] minimum`, `maximum` | number | yes | | Soft limits; `0 <= minimum < maximum <= 1` |
| `[flats] <name>` | string | no | no flats | File beside the TOML: `.npy` surface or `.txt` vector (see 4.4) |
| `[sdk] profiles_path` | string | no | the SDK's default (`/opt/Boston Micromachines/Profiles`) | Folder holding `<serial>.dm` |
| `[sdk] maps_path` | string | no | the SDK's default (`/opt/Boston Micromachines/Map`) | Folder holding BMC's `.map` files |
| `[log] file` | string | no | `bmc_dm.log` in the working folder | Log file; its folder is created if missing |

Rules:

- Unknown sections or keys, and values of the wrong type, are errors (catches typos).
- Numbers may be written as integers (`maximum = 1`); booleans are not numbers.
- Relative paths resolve against the TOML's folder; `~` expands to the home folder.
- Leave out `[sdk]` with a standard SDK install.

### 4.3 Actuator map (`actuator_map.txt`)

A square grid of integers, whitespace separated, one grid row per line. Each cell holds
the vector index of the actuator at that position, or `-1` where there is no actuator.
Every index from 0 to N-1 must appear exactly once. Lines starting with `#` are comments.

```
# 6x6 example: 32 actuators, numbered row by row
  -1    0    1    2    3   -1
   4    5    6    7    8    9
  10   11   12   13   14   15
  ...
```

BMC's circular kilo-DM layouts (648 actuators on 28x28, 952 on 34x34) can be generated:

```python
import numpy as np
from bmc_dm import ActuatorMap

np.savetxt("actuator_map.txt", ActuatorMap.circular(28).indices, fmt="%4d")
```

Row 0 of the file is row 0 of every surface. How that grid appears on a camera
(flips or rotations) depends on the optics; check it with `letter_f`.

### 4.4 Flats

| File | Contents |
| --- | --- |
| `.npy` | A surface: a `side x side` numpy array of finite values. Values in cells without an actuator are ignored |
| `.txt` | A vector: `num_actuators` numbers in actuator order (BMC's text format), whitespace or newline separated |

Flats are read-only once loaded. Values above the soft limits are clipped when applied
(with a warning). To try a flat that is not in the bundle, load it yourself and send it:
`dm.set_surface(np.load("other_flat.npy"))`.

### 4.5 Last command (`last_command.npy`)

- Written after every command the driver sends, including each step of `flash_edge` and
  `simulate_turbulence`: a vector of `num_actuators` values, after clipping.
- Written to a temporary file first and then renamed, so a crash never leaves a partial
  file. If it cannot be written, the command is still sent and a warning is logged.
- Read by `connect()`. A missing file means "no previous command". An unreadable or
  invalid file (wrong length, NaN, not a numpy file) is ignored with a warning.
- Shared by every TOML in the same folder. Delete it to forget the last shape.
- Git-ignored, so runtime state is never committed.

### 4.6 Building a config in code

`DMConfig` holds a loaded bundle and can also be built directly:

```python
from pathlib import Path
from bmc_dm import ActuatorMap, BmcDmController, DMConfig

config = DMConfig(
    name="Test DM",
    actuator_map=ActuatorMap.circular(28),
    soft_limits=(0.0, 0.8),
    serial_number="30CW009#052",  # optional
    flats={"inst": my_flat_surface},  # optional, name to surface
    default_flat="inst",  # optional, no automatic default here
    sdk_profiles_path=None,  # optional Path
    sdk_maps_path=None,  # optional Path
    logfile="bmc_dm",  # ".log" is appended when missing
    last_command_path=Path("last_command.npy"),  # optional; None saves nothing
)
dm = BmcDmController(config)
```

A config built this way saves no last command unless `last_command_path` is given.
`dataclasses.replace` derives a variant of a loaded bundle, for example
`replace(DMConfig.load("configs/sight/sight.toml"), last_command_path=None)`.

## 5. Controller reference

```python
from bmc_dm import BmcDmController

dm = BmcDmController(config, log=True)
```

| Argument | Meaning |
| --- | --- |
| `config` | Path to the bundle's TOML file (string or `Path`), or a `DMConfig` |
| `log` | `True`: log to the console and the bundle's log file. `False`: print messages instead |

Creating a controller loads and checks the bundle only; it never touches the SDK.

### 5.1 Attributes

| Attribute | Meaning |
| --- | --- |
| `dm.config` | The loaded `DMConfig`: `name`, `soft_limits`, `serial_number`, `flats` (name to surface), `default_flat`, paths |
| `dm.actuator_map` | The bundle's `ActuatorMap` (section 7) |
| `dm.flat` | The default flat surface (not applied), or `None` without one |
| `dm.lock` | Reentrant lock held around every SDK call; hold it yourself to group calls (see 5.7) |

### 5.2 Connection

| Method | What it does |
| --- | --- |
| `connect(serial_number=None)` | Opens the mirror (serial from the argument, else the bundle). Sets the SDK paths from `[sdk]`, checks that the mirror's actuator count matches the map, and loads `last_command.npy`. Sends no command |
| `disconnect()` | Closes the mirror. Sends no command, but the SDK zeroes the mirror as it closes. A second call only logs a warning |
| `restore()` | Sends the last command again, typically right after `connect()`. Returns the applied surface |
| `is_connected()` | `True` while connected |
| `with BmcDmController(...) as dm:` | Disconnects on leaving the block, even after an error. `connect()` is still called explicitly inside |

`connect()` raises `DMError` if the SDK module is missing, the DM is already connected,
the SDK reports an error (for example, no profile for that serial), or the actuator
count differs from the map. On a count mismatch it closes the mirror it opened, which
zeroes that mirror. It raises `ValueError` if no serial number is known.

### 5.3 Sending commands

Every method below sends one command (unless noted), saves it as the last command, and
returns the surface that was applied, after clipping. All of them need a connected DM.

| Method | Sends |
| --- | --- |
| `set_vector(vector)` | One value per actuator, in actuator order |
| `set_surface(surface)` | A `side x side` surface; cells without an actuator are ignored |
| `set_flat(name=None)` | A flat from the bundle, by default the default flat |
| `zero()` | 0 on every actuator |
| `initialize()` | The default flat, or zeros without one; returns `True` and marks the DM initialized (`is_initialized()`) |
| `poke(pattern, amplitude, *, bias=None, **pattern_args)` | `bias + amplitude x pattern` (see 5.4) |
| `add(pattern, amplitude, **pattern_args)` | `last command + amplitude x pattern` |

Before sending, every command is checked: a wrong shape, non-numeric values, or NaN or
infinite values raise `ValueError` and nothing is sent.

### 5.4 Pokes and patterns

`poke` takes a pattern in either of two forms:

- a pattern function from section 6, plus its arguments as keywords; the driver supplies
  the actuator map: `dm.poke(zernike, 0.2, noll=5)`;
- a ready-made surface: `dm.poke(crosshair(dm.actuator_map, width=4), -0.1)`.

`bias` sets what the pattern is added to:

| `bias` | Base |
| --- | --- |
| `None` (default) | The default flat, or zeros without one |
| a number, e.g. `0` or `0.4` | That value on every actuator (`bias=0` starts from zeros) |
| a surface | That surface |

`amplitude` is in normalized units. Patterns are unit sized: selections are 1 on the
chosen actuators and 0 elsewhere, `sine` runs from -1 to 1, and Zernikes have a
peak-to-valley of 1. So `dm.poke(actuators, -0.1, indices=0)` lowers actuator 0 by 0.1
below the flat.

`add` works like `poke` with the last command as the bias. Right after `connect()` that is
the command loaded from `last_command.npy`.

```python
dm.poke(crosshair, -0.1)  # flat - 0.1 on a centered cross
dm.poke(grid, 0.1, row=0, column=0, spacing=4)
dm.poke(sine, 0.05, frequency=3, angle=0.5)  # 3 cycles across the grid, tilted 0.5 rad
dm.poke(zernike, 0.2, noll=5, bias=0)  # on zeros instead of the flat
dm.add(actuators, 0.02, indices=[100, 101])  # nudge two actuators
```

### 5.5 Reading the shape

| Method | Returns |
| --- | --- |
| `get_vector()` | The last command, one value per actuator |
| `get_surface()` | The last command as a surface |

The last command is the one this session sent last, or right after `connect()` the one
loaded from `last_command.npy` (zeros without one). It is the driver's record of what it
commanded, not a measurement: the SDK has no way to read the mirror's real shape, so
drift can only be seen optically. Right after `connect()` the mirror has usually been
zeroed by the previous session's close; `restore()` puts the shape back. Both methods
raise `DMError` when not connected.

### 5.6 Routines

**`flash_edge(amplitude=-0.2, pause_s=0.3, iterations=20, *, bias=None)`** alternates two
interleaved sets of edge actuators (every third edge actuator, starting at index 0 and at
index 1), holding each for `pause_s`, so the pupil edge blinks on a camera. It blocks for
about `2 x pause_s x iterations` seconds (12 s by default), then sends `bias` (as in
`poke`: the default flat by default). The bias is also restored after Ctrl-C or an
error, and the Ctrl-C or error is then raised as usual.

**`simulate_turbulence(nolls, amplitudes=0.002, rate_hz=5.0, duration_s=10.0, *, bias=None)`**
sends `rate_hz x duration_s` random wavefronts on top of `bias`, each held for
`1 / rate_hz` seconds. Every wavefront is a sum of the Zernikes in `nolls`, each with an
amplitude drawn from a zero-mean Gaussian whose standard deviation is `amplitudes` (one
number for all, or one per Noll index). Ctrl-C stops it early without raising. The bias is
sent again at the end. It returns the wavefronts actually applied, without the bias, as
an array of shape `(steps, side, side)`. Needs poppy.

```python
dm.flash_edge(-0.1, pause_s=0.3, iterations=20)
wavefronts = dm.simulate_turbulence(range(4, 30), amplitudes=0.002, rate_hz=5, duration_s=10)
```

### 5.7 Status, logging and threads

- `get_status()` returns `(code, message)` for the latest message: code `-1` after an
  error, `0` otherwise.
- `set_verbose(True)` also logs debug messages, such as `flash_edge` progress.
- Every SDK call runs under `dm.lock`, so several threads may share one controller. Hold
  the lock to make several calls one step, for example reading then sending:

  ```python
  with dm.lock:
      surface = dm.get_surface()
      dm.set_surface(surface * 0.9)
  ```

## 6. Pattern functions

Each function takes the actuator map first and returns a `side x side` surface. Call them
directly as `crosshair(dm.actuator_map)`, or pass them to `poke`/`add` without the map:
`dm.poke(crosshair, -0.1, width=4)`.

| Function | Returns | Errors |
| --- | --- | --- |
| `actuators(map, indices)` | 1 on the actuators with these vector indices (an int or a list, counted from 0) | `TypeError` for non-integers, `ValueError` out of range |
| `grid(map, row, column, spacing)` | 1 on one cell per `spacing x spacing` block, at `(row, column)` within each block | `ValueError` if `row` or `column` is negative or `spacing < 1` |
| `rows(map, indices)` | 1 on whole grid rows (an int or a list) | numpy's `IndexError` out of range |
| `columns(map, indices)` | 1 on whole grid columns | numpy's `IndexError` out of range |
| `crosshair(map, width=2)` | 1 on a centered cross with bars `width` cells wide (on an even grid like Sight's, an even width centers it exactly) | `ValueError` if `width < 1` |
| `letter_f(map)` | 1 on a capital F, scaled to the grid, to check orientation on a camera | |
| `edge(map, start=0, step=1)` | 1 on every `step`-th edge actuator, among edge actuators with index `>= start` | |
| `sine(map, frequency=1.0, angle=0.0, phase=0.0)` | A sinusoid from -1 to 1 with `frequency` cycles across the grid, at `angle` and `phase` in radians | |
| `zernike(map, noll)` | One Zernike by Noll index (1 is piston, 4 focus, 5 and 6 astigmatism) | needs poppy |
| `zernike_basis(map, max_noll)` | Array `(max_noll, side, side)`; `basis[noll - 1]` is one Zernike | `ValueError` if `max_noll < 1`; needs poppy |

Notes:

- *Edge actuators* are those missing a neighbor above, below, left or right (80 on
  Sight). With row-by-row numbering, consecutive edge indices alternate between the left
  and right sides, so use an odd `step` with `edge` to spread the selection around.
- *Zernikes* come from poppy, on a circle that covers the active region. Every term except
  piston is scaled to a peak-to-valley of 1 over the actuators. Bases are cached, so
  repeated calls are fast. Without poppy, they raise `ModuleNotFoundError`; install it
  with `pip install -e ".[zernike]"`.
- Selections and the F may cover cells without actuators; those cells are ignored when
  sent.

## 7. ActuatorMap

`dm.actuator_map` (or `ActuatorMap(...)`, `ActuatorMap.circular(side)`,
`ActuatorMap.from_file(path)`) converts between vectors and surfaces:

| Member | Meaning |
| --- | --- |
| `side`, `shape` | Grid size, and `(side, side)` |
| `num_actuators` | Number of actuators N |
| `indices` | The grid of vector indices, `-1` where there is no actuator |
| `mask` | `True` on cells with an actuator |
| `edge_actuators` | Sorted vector indices of the edge actuators |
| `to_surface(vector)` | Place N values onto the grid (0 elsewhere) |
| `to_vector(surface)` | Read N values off a surface, in actuator order |
| `circular(side)` | BMC's circular layout, numbered row by row (648 at 28, 952 at 34) |
| `from_file(path)` | Read an `actuator_map.txt` |

All arrays are read-only. A grid that is not square or does not number 0 to N-1 exactly
once raises `ValueError`.

```python
amap = dm.actuator_map
print(amap.indices[0])  # vector indices along grid row 0
row, column = np.argwhere(amap.indices == 390)[0]  # where actuator 390 sits
```

## 8. Errors and logging

| Exception | Raised for |
| --- | --- |
| `DMError` | SDK failures (with BMC's own message), commands or reads while not connected, a missing SDK, an actuator count mismatch |
| `ConfigError` | An invalid bundle: TOML, actuator map or flat. Subclass of `ValueError`; the message names the file |
| `ValueError` | Bad input: wrong shape, NaN or infinite values, unknown flat, bad routine arguments, no serial number |
| `TypeError` | Non-integer actuator indices |

Controller errors are also logged and set `get_status()`. A command that fails its
checks is not sent.

```python
from bmc_dm import DMError

try:
    dm.set_flat()
except DMError as error:
    print(f"DM failed: {error}")
```

Logging (`log=True`, the default) writes to the console and to the bundle's log file
(`[log] file`, else `bmc_dm.log` in the working folder). Messages include connects and
disconnects, clipping warnings, loaded or ignored last commands, and all errors. The SDK
also prints its own `DEBUG - ...` and `WARN - ...` lines to the console; they are normal.

## 9. How the SDK behaves

These are properties of BMC's DM-SDK, confirmed in its source, that shape how the driver
works:

- **Closing zeroes the mirror.** The SDK's `close_dm()` sets every actuator to zero and
  disables the outputs (USB drivers also switch off the high voltage). The SDK does the
  same when its object is destroyed, including when Python exits. A mirror keeps its
  shape only while a controller stays connected; `last_command.npy` and `restore()`
  bring it back in the next session.
- **The mirror cannot be read back.** The SDK only sends values to the hardware. Its
  `get_actuator_data()` returns its own copy of the last command (zeros after opening).
  The driver keeps its own record instead (section 5.5).
- **SDK calls on a closed mirror crash Python** (segmentation fault). The driver checks
  the connection before every call, so this cannot happen through the driver; do not
  call the `bmc` module directly on a closed mirror.
- **Serial numbers are exactly 11 characters**, e.g. `30CW009#052`. Others fail with
  "An invalid serial number was entered."
- **A missing profile** fails with "Error opening file." The SDK looks for
  `<serial>.dm` in its profiles folder (`[sdk] profiles_path` to change it).

## 10. The Sight bundle

`configs/sight/` describes the Sight testbed's mirror:

| | |
| --- | --- |
| Mirror | BMC kilo-DM, serial `30CW009#052` |
| Grid | 28 x 28, 648 actuators in a circle, numbered row by row (`ActuatorMap.circular(28)`), 80 edge actuators |
| Soft limits | 0.0 to 0.8 |
| Default flat | `inst` |

| Flat | Original file |
| --- | --- |
| `inst` | `Sight_INST_flat.npy` |
| `quick` | `SightDM_2024Oct4_QuickFlat.npy` |
| `mean250`, `mean400`, `mean500`, `mean600`, `mean700` | `SightDM_2024Oct7_Mean250.npy` and so on |

`mean700` peaks at 0.9, so applying it clips 623 of its 648 values to 0.8 and logs a
warning; that is expected.

Actuator indices are counted from 0, as in the lab's original code. Patterns match that
code exactly, except `letter_f`, which is centered on the grid.

## 11. Trying it without a mirror

The SDK has a dummy driver for software testing: a profile with `type=2` opens without
hardware and accepts every command. Use it under a made-up serial and a separate
profiles folder, so it can never be mistaken for the real mirror's profile:

```python
from dataclasses import replace
from pathlib import Path

from bmc_dm import BmcDmController, DMConfig, zernike

profiles = Path("dummy_profiles")
profiles.mkdir(exist_ok=True)
lines = ["type=2", "hva=0", "size=648", "maxv=300", "vlim=150", "map=", "inactive=0"]
# BMC profiles are text with every byte XOR-ed with 0x83
(profiles / "DUMMY000648.dm").write_bytes(
    b"\n".join(bytes(b ^ 0x83 for b in line.encode()) for line in lines) + b"\n"
)

config = replace(
    DMConfig.load("configs/sight/sight.toml"),
    sdk_profiles_path=profiles.resolve(),
    last_command_path=None,  # keep the dry run out of the bundle's last_command.npy
)
with BmcDmController(config) as dm:
    dm.connect("DUMMY000648")
    dm.set_flat()
    dm.poke(zernike, 0.1, noll=5)
```

Set `size` to your mirror's actuator count. Every driver feature runs this way; only the
SDK's own readback (`get_actuator_data()`) stays at zero.

## 12. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `DMError: BMC DM-SDK Python module (bmc) is not installed` | `import bmc` failed. Run `python -c "import bmc"` to see why, then use the rows below |
| `No module named 'bmc'` | The venv cannot see the SDK. Add the `.pth` file (section 1.2), or install the SDK (1.1) |
| `libBMC.so.4: cannot open shared object file` | The SDK's libraries are not registered. Rerun `sudo ./install.sh`, or set `LD_LIBRARY_PATH="/opt/Boston Micromachines/lib"` |
| `import bmc` crashes or reports an undefined symbol | Most likely the venv's Python differs from the one the SDK was built for. Create the venv with the system `python3` |
| `Failed to open DM ...: Error opening file.` | No `<serial>.dm` in the profiles folder. Copy BMC's profile (1.1 step 5) or set `[sdk] profiles_path` |
| `Failed to open DM ...: An invalid serial number was entered.` | The serial is not 11 characters. Check `[dm] serial_number` or the `connect()` argument |
| `DM ... has N actuators, the ... actuator map has M` | The bundle does not match the mirror (or the profile). Use the right bundle |
| Opening a USB DM fails with a permissions or access error | udev rules not applied. Unplug and replug the driver after installing |
| `ModuleNotFoundError: No module named 'poppy'` | Zernikes and turbulence need poppy: `pip install -e ".[zernike]"` |
| `Clipped N of 648 values to [0.0, 0.8]` | The command went past the soft limits; the clipped command was sent. Lower the amplitude or bias |
| `Ignoring saved command ...` | `last_command.npy` was invalid and was ignored; the next command overwrites it |
| The mirror is flat after `connect()` | Expected: the SDK zeroed it when the last session closed. Call `restore()` |
| SDK build fails at the SWIG or Python step | The source folder path contains a space. Move it to a path without spaces |
| PCIe `modprobe` fails: "Key was rejected by service" | Secure Boot blocks the module. Disable Secure Boot or sign the module |

## 13. Development

The tests replace the SDK with a fake `bmc` module, so they run on any machine without
the SDK or a mirror:

```bash
pip install -e ".[dev,zernike]"
pytest && mypy bmc_dm.py test_bmc_dm.py && pylint bmc_dm.py test_bmc_dm.py \
  && ruff check . && ruff format --check .
```

The whole driver is one file, `bmc_dm.py`; the tests are in `test_bmc_dm.py`.
