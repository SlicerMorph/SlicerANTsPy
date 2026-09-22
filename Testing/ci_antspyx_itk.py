"""EXPERIMENT (throwaway branch): which package makes Slicer abort on Linux?

Issue #29 / PR #30, November 2025: Slicer aborted on Linux once this extension's
dependencies were installed, reporting two ITK versions in one process.  The fix was a
locally built antspyx, linked against Slicer's ITK 5.4.4, served from a Box share -- and
that is still what the extension downloads on Linux, unchanged since 2025-11-10.

Two runs on 2026-09-22 say the story is not what it looks like:

* The production path (itk-ants + the Box antspyx) still aborts, on Slicer 5.12.4 AND
  preview: every check passes, then `SlicerApp-real exit abnormally` during shutdown.
* antspyx ALONE does not abort -- neither the Box wheel (ITK 5.4.4) nor the PyPI one
  (ITK 5.4.5) -- although neither matches Slicer's 5.4.6.

So a mismatched antspyx is survivable, and the Box wheel may be fixing nothing.  The
remaining suspect is ``itk-ants``, which pip-installs the ``itk`` wheel: a THIRD ITK in
the process.  This installs each combination on its own and reports which ones abort.

COMBO picks the combination:

    itkants        itk-ants only
    box            the Box antspyx only (production path on Linux)
    pypi           the PyPI antspyx only
    itkants+box    both -- what a user actually ends up with today
    itkants+pypi   both, with the PyPI wheel instead

The abort happens during shutdown, after the checks have all passed, so the checks are
not the result: the job's exit code is.  A job that reaches "[ci] ... checks passed" and
then fails anyway is one where Slicer could not shut down cleanly.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_common  # noqa: E402

import slicer  # noqa: E402

COMBO = os.environ.get("COMBO", "itkants+box")

# The dependency set the extension installs alongside antspyx, minus scipy, whose
# `scipy<1.16` pin would downgrade Slicer's own.
ANTSPYX_DEPS = "pandas pyyaml statsmodels webcolors matplotlib scikit-learn"

_ITK_VERSION = re.compile(rb"itk version (\d+\.\d+\.\d+(?:\.post\d+)?)")


def itkVersionsIn(path):
    """Every ITK version baked into a binary, found by scanning it for the literal ITK
    itself prints ("itk version 5.4.6") -- more portable here than asking each library
    through an API it may not expose."""
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
    for name, versions in sorted(found.items())[:6]:
        ci_common.say(f"[ci]   {label}: {name} -> {', '.join(versions)}")
    allVersions = sorted({v for versions in found.values() for v in versions})
    ci_common.say(f"[ci] ITK in {label}: {', '.join(allVersions) or '(none)'}")
    return allVersions


def packageDirectory(moduleName):
    try:
        module = __import__(moduleName)
        return os.path.dirname(module.__file__)
    except Exception:
        return None


def main():
    ci_common.say(f"[ci] COMBO={COMBO}")
    home = slicer.app.slicerHome

    versions = {"Slicer": reportItkVersions("Slicer", [
        os.path.join(home, "lib", "**", "*ITKCommon*")])}

    # --- install exactly what this combination calls for ----------------------
    if "itkants" in COMBO:
        import ITKANTsCommon
        # confirm=False: the installer would otherwise wait on a modal dialog.
        ITKANTsCommon.ITKANTsCommonLogic.installITK(confirm=False)
        import itk  # noqa: F401
        ci_common.record("itk-ants installs and imports", True)
        versions["itk (pip)"] = reportItkVersions("itk (pip)", [
            os.path.join(packageDirectory("itk") or "", "**", "*.so")])

    if "box" in COMBO:
        import ANTsPyRegistration
        ANTsPyRegistration.ANTsPyRegistrationLogic().installANTsPyX()
        ci_common.record("antspyx installs (Box wheel)", True)
    elif "pypi" in COMBO:
        slicer.util.pip_install("antspyx --no-deps")
        slicer.util.pip_install(ANTSPYX_DEPS)
        ci_common.record("antspyx installs (PyPI wheel)", True)

    if "box" in COMBO or "pypi" in COMBO:
        import ants
        ci_common.record("antspyx imports", True, f"version {ants.__version__}")
        versions["antspyx"] = reportItkVersions("antspyx", [
            os.path.join(os.path.dirname(ants.__file__), "**", "*.so")])

    distinct = sorted({v for found in versions.values() for v in found})
    ci_common.say(f"[ci] ITK builds in this process: {len(distinct)} -> {', '.join(distinct)}")
    for label, found in versions.items():
        ci_common.say(f"[ci]   {label}: {', '.join(found) or '(none)'}")

    # --- exercise both sides in the one process -------------------------------
    import numpy as np

    volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "probe")
    slicer.util.updateVolumeFromArray(
        volume, np.random.default_rng(0).random((24, 24, 24)).astype("float32"))
    path = os.path.join(slicer.app.temporaryPath, "itk-probe.nrrd")
    ci_common.record("Slicer writes a volume through ITK", slicer.util.saveNode(volume, path))
    ci_common.record("Slicer reads it back through ITK",
                     slicer.util.loadVolume(path) is not None)

    if "box" in COMBO or "pypi" in COMBO:
        import ants
        fixed = ants.from_numpy(np.random.default_rng(1).random((24, 24, 24)))
        moving = ants.from_numpy(np.random.default_rng(2).random((24, 24, 24)))
        result = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        ci_common.record("ants.registration completes", "warpedmovout" in result)

    ci_common.record("Slicer still does ITK IO afterwards",
                     slicer.util.loadVolume(path) is not None)
    ci_common.say("[ci] work finished; whether this job passes now depends on shutdown")


ci_common.run(main, f"itk-combo[{COMBO}]")
