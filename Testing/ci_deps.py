"""Install this extension's runtime dependencies on every platform, then import them.

Run INSIDE Slicer by ``.github/workflows/slicer-deps.yml`` (weekly, and on demand).

SlicerANTsPy does not ship its heavy dependencies: the first time a user opens the
module it downloads and pip-installs them into Slicer's own Python.  That step is
the most platform-specific code in the extension and the least visible when it
breaks -- it fails on someone else's operating system, weeks later, with no error
anyone here ever sees.  Specifically:

* ``ITKANTsCommonLogic.installITK`` installs ``itk-ants`` from PyPI, which must
  publish a wheel for this platform AND this Python version.
* ``ANTsPyRegistrationLogic.installANTsPyX`` installs ``antspyx`` with
  ``--no-deps`` (its scipy pin would otherwise downgrade Slicer's scipy and break
  the application), then adds the remaining dependencies by hand.  On Linux there
  is no usable PyPI wheel, so it downloads one from a fixed Box share whose
  filename hard-codes ``cp312`` -- that URL is a third-party dependency that can
  disappear, and the filename silently assumes Slicer's Python stays at 3.12.

So this calls the extension's own installer functions -- not a hand-written pip
command -- then imports the packages, because a wheel that installs and cannot be
imported is a real failure mode, and finally runs an actual
``antsRegistrationSyNQuick[s]`` registration, because a wheel that imports and
cannot register is another one.

This job touches the network and downloads hundreds of megabytes, which is why it
runs on a schedule rather than on every pull request: a PyPI or Box outage should
not red out someone's PR.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_common  # noqa: E402

import slicer  # noqa: E402


def timed(step):
    """Run `step`, returning (ok, detail) with the elapsed time or the first error line."""
    started = time.time()
    try:
        step()
        return True, f"{time.time() - started:.0f}s"
    except Exception as error:
        return False, f"after {time.time() - started:.0f}s: " \
                      f"{type(error).__name__}: {ci_common.firstLine(error)}"


SHAPE = (48, 48, 48)
TRANSLATION = (0.0, 4.0, 3.0)   # voxels; what the registration has to recover


def texturedVolume(seed=0, shift=None):
    """Band-limited noise: a well-posed registration target.

    The first version of this check used a single Gaussian blob, and it was a bad
    test.  A smooth, symmetric blob gives the similarity metric almost no gradient
    to follow, so the result is decided by numerical noise: with identical input,
    two platforms improved alignment and four made it worse, each in about a second.
    That measured the test, not the wheel.

    Smoothed random noise has texture everywhere, so recovering a pure translation
    is unambiguous and every correct build should manage it.
    """
    import numpy as np
    from scipy import ndimage

    volume = ndimage.gaussian_filter(
        np.random.default_rng(seed).random(SHAPE).astype("float32"), sigma=2.0)
    volume -= volume.min()
    volume /= max(float(volume.max()), 1e-8)
    if shift is not None:
        volume = ndimage.shift(volume, shift, order=3, mode="nearest")
    return volume.astype("float32")


def runRegistration():
    """Register a displaced blob onto a fixed one with antsRegistrationSyNQuick[s].

    Installing and importing antspyx only proves the wheel is present and loadable.
    This runs the deformable pipeline users actually run -- rigid, then affine, then
    SyN -- which is what exercises the compiled ANTs code, the ITK inside it, and the
    threading, in the same process as Slicer.  Whether the registration is any good is
    not the question; that it completes and moves the moving image toward the fixed one
    is, so the check is that the mean squared difference goes down.  A registration that
    silently makes alignment worse is a broken build, not a tuning problem.
    """
    import ants
    import numpy as np

    fixedArray = texturedVolume()
    movingArray = texturedVolume(shift=TRANSLATION)

    result = ants.registration(
        fixed=ants.from_numpy(fixedArray),
        moving=ants.from_numpy(movingArray),
        type_of_transform="antsRegistrationSyNQuick[s]")

    warpedArray = result["warpedmovout"].numpy()
    before = float(np.mean((fixedArray - movingArray) ** 2))
    after = float(np.mean((fixedArray - warpedArray) ** 2))
    ci_common.say(f"[ci] mean squared difference {before:.5g} -> {after:.5g} "
                  f"({before / max(after, 1e-12):.1f}x better, "
                  f"{len(result['fwdtransforms'])} forward transforms)")
    # A generous margin: recovering a pure translation from a textured image should
    # remove most of the difference, so this fails only on a genuinely broken build,
    # not on ordinary numerical variation between platforms.
    if not after < 0.5 * before:
        raise RuntimeError(
            f"registration barely moved the image: {before:.5g} -> {after:.5g}")


def main():
    say = ci_common.say

    # --- itk-ants, via the extension's own installer -------------------------
    import ITKANTsCommon
    # confirm=False: the installer would otherwise ask for confirmation in a modal
    # dialog, which never gets answered in a headless run.
    ok, detail = timed(lambda: ITKANTsCommon.ITKANTsCommonLogic.installITK(confirm=False))
    ci_common.record("itk-ants installs", ok, detail)

    if ok:
        def importItk():
            import itk
            itk.ANTSRegistration  # forces the itk-ants shared library to load
        ok, detail = timed(importItk)
        ci_common.record("itk-ants imports and loads its library", ok, detail)

    # --- antspyx, via the extension's own installer --------------------------
    import ANTsPyRegistration
    logic = ANTsPyRegistration.ANTsPyRegistrationLogic()
    ok, detail = timed(logic.installANTsPyX)
    ci_common.record("antspyx installs", ok, detail)

    if ok:
        def importAnts():
            import ants
            say(f"[ci] antspyx version: {ants.__version__}")
        ok, detail = timed(importAnts)
        ci_common.record("antspyx imports", ok, detail)

    if ok:
        ok, detail = timed(runRegistration)
        ci_common.record("antsRegistrationSyNQuick[s] runs", ok, detail)

    # --- Slicer must still be usable afterwards ------------------------------
    # antspyx pins scipy<1.16 while Slicer bundles a newer one; the installer
    # works around that with --no-deps.  If that workaround ever stops holding,
    # the dependencies pulled in afterwards downgrade scipy and break Slicer
    # itself -- a far worse outcome than a failed install, and invisible unless
    # it is checked explicitly.
    try:
        import scipy
        import numpy
        say(f"[ci] after install: scipy={scipy.__version__} numpy={numpy.__version__}")
        ci_common.record("Slicer's scipy/numpy still import", True,
                         f"scipy {scipy.__version__}")
    except Exception as error:
        ci_common.record("Slicer's scipy/numpy still import", False,
                         f"{type(error).__name__}: {ci_common.firstLine(error)}")


ci_common.run(main, "deps")
