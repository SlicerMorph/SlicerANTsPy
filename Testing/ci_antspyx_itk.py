"""EXPERIMENT (throwaway branch): does the PyPI antspyx wheel still clash with Slicer's ITK?

Background.  Issue #29 / PR #30, November 2025: the PyPI antspyx wheel linked ITK 5.4.3
while Slicer linked 5.4.4.post1, two different ITK builds ended up in one process, and
Slicer aborted on Linux -- ELF resolves the duplicate symbols into one another, which
macOS's two-level namespace does not.  The workaround was a locally built antspyx,
linked against 5.4.4, served from a Box share.  That is still what the extension
downloads on Linux today, unchanged since 2025-11-10.

Since then both sides moved and neither one landed where the other is:

    Slicer 5.12.4          ITK 5.4.6
    PyPI antspyx 0.6.2/3   ITK 5.4.5   (ANTsPy scripts/configure_ITK.sh)
    the Box wheel          ITK 5.4.4   (per PR #30)

So the question is not "is there a prebuilt wheel" -- there has always been a
manylinux_2_17 one -- but "which wheel, if either, survives in the same process as
Slicer's ITK".  ANTSPYX_SOURCE picks which one this run installs:

    box   -- the production path, logic.installANTsPyX()
    pypi  -- plain `pip install antspyx --no-deps`, the wheel the workaround replaced

Then it does what actually crashed: ITK work on the Slicer side, ants work on the other,
in one process.  A clash aborts the application, so the job dies and the log stops --
that IS the result.  Reaching the summary line means the combination survived.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_common  # noqa: E402

import slicer  # noqa: E402

SOURCE = os.environ.get("ANTSPYX_SOURCE", "box")

# The same dependency set the extension installs alongside antspyx, minus scipy: its
# `scipy<1.16` pin would downgrade Slicer's own scipy.
ANTSPYX_DEPS = "pandas pyyaml statsmodels webcolors matplotlib scikit-learn"

_ITK_VERSION = re.compile(rb"itk version (\d+\.\d+\.\d+(?:\.post\d+)?)")


def itkVersionsIn(path):
    """Every ITK version string baked into a binary, by scanning it for the literal
    ITK itself prints ("itk version 5.4.6").  Cheaper and more portable here than
    getting each library to report its own version through an API it may not expose."""
    try:
        with open(path, "rb") as fp:
            return sorted({match.decode() for match in _ITK_VERSION.findall(fp.read())})
    except Exception:
        return []


def reportItkVersions(label, patterns):
    found = {}
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            versions = itkVersionsIn(path)
            if versions:
                found[os.path.basename(path)] = versions
    for name, versions in sorted(found.items()):
        ci_common.say(f"[ci]   {label}: {name} -> {', '.join(versions)}")
    allVersions = sorted({v for versions in found.values() for v in versions})
    ci_common.say(f"[ci] {label} ITK: {', '.join(allVersions) or '(none found)'}")
    return allVersions


def main():
    ci_common.say(f"[ci] ANTSPYX_SOURCE={SOURCE}")
    home = slicer.app.slicerHome

    slicerItk = reportItkVersions("Slicer", [
        os.path.join(home, "lib", "**", "*ITKCommon*"),
        os.path.join(home, "**", "*ITKCommon*"),
    ])
    ci_common.record("Slicer's ITK version is detectable", bool(slicerItk),
                     ", ".join(slicerItk))

    # --- install antspyx the way this variant is supposed to ------------------
    if SOURCE == "box":
        import ANTsPyRegistration
        logic = ANTsPyRegistration.ANTsPyRegistrationLogic()
        try:
            logic.installANTsPyX()
            ci_common.record("antspyx installs (Box wheel, production path)", True)
        except Exception as error:
            ci_common.record("antspyx installs (Box wheel, production path)", False,
                             f"{type(error).__name__}: {ci_common.firstLine(error)}")
            return
    else:
        try:
            slicer.util.pip_install("antspyx --no-deps")
            slicer.util.pip_install(ANTSPYX_DEPS)
            ci_common.record("antspyx installs (PyPI manylinux wheel)", True)
        except Exception as error:
            ci_common.record("antspyx installs (PyPI manylinux wheel)", False,
                             f"{type(error).__name__}: {ci_common.firstLine(error)}")
            return

    try:
        import ants
        ci_common.record("antspyx imports", True, f"version {ants.__version__}")
    except Exception as error:
        ci_common.record("antspyx imports", False,
                         f"{type(error).__name__}: {ci_common.firstLine(error)}")
        return

    antsItk = reportItkVersions("antspyx", [
        os.path.join(os.path.dirname(ants.__file__), "**", "*.so"),
        os.path.join(os.path.dirname(ants.__file__), "**", "*.dylib"),
    ])
    ci_common.record("antspyx's ITK version is detectable", bool(antsItk),
                     ", ".join(antsItk))
    matched = bool(slicerItk) and bool(antsItk) and set(slicerItk) == set(antsItk)
    ci_common.say(f"[ci] ITK match: Slicer {slicerItk} vs antspyx {antsItk} -> "
                  f"{'same' if matched else 'DIFFERENT'}")

    # --- the part that actually crashed --------------------------------------
    # Slicer-side ITK first (its image IO factories are what reported the clash), then
    # ants in the same process, then Slicer-side ITK again.  If the two ITKs are
    # incompatible the application aborts here and this script never reaches its end.
    import numpy as np

    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "probe")
    slicer.util.updateVolumeFromArray(
        volume, np.random.default_rng(0).random((24, 24, 24)).astype("float32"))
    path = os.path.join(slicer.app.temporaryPath, "itk-probe.nrrd")
    ci_common.record("Slicer writes a volume through ITK", slicer.util.saveNode(volume, path))
    ci_common.record("Slicer reads it back through ITK",
                     slicer.util.loadVolume(path) is not None)

    ci_common.say("[ci] running an ants registration in the same process ...")
    fixed = ants.from_numpy(np.random.default_rng(1).random((24, 24, 24)))
    moving = ants.from_numpy(np.random.default_rng(2).random((24, 24, 24)))
    result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
    ci_common.record("ants.registration completes", "warpedmovout" in result,
                     f"keys: {sorted(result)[:3]}")

    ci_common.record("Slicer still does ITK IO afterwards",
                     slicer.util.loadVolume(path) is not None)


ci_common.run(main, f"antspyx-itk[{SOURCE}]")
