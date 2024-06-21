import logging
import os

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
)


class ITKANTsCommon(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ITK ANTs Common Utilities"
        self.parent.categories = ["Registration"]
        self.parent.dependencies = []
        self.parent.contributors = ["Dženan Zukić (Kitware Inc.)"]
        self.parent.helpText = (
            "This is a helper module, which contains commonly used ITK functions."
        )
        self.parent.acknowledgementText = """
This file was originally developed by Dženan Zukić, Kitware Inc.,
and was partially funded by NIH grant 5R44CA239830.
"""
        self.parent.hidden = True
        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", preloadITK)


class ITKANTsCommonLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._itk = None

    @property
    def itk(self):
        if self._itk is None:
            logging.info("Importing itk...")
            self._itk = self.importITK()
        return self._itk

    def importITK(self, confirmInstallation=True):
        perform_install = False
        try:
            import itk
        except ModuleNotFoundError:
            perform_install = True
        if not hasattr(itk, 'ANTSRegistration'):
            perform_install = True
        if perform_install:
            with slicer.util.WaitCursor(), slicer.util.displayPythonShell():
                itk = self.installITK(confirmInstallation)
                if itk is None:
                    return None
        logging.info(f"ITK {itk.__version__} imported correctly")
        return itk

    @staticmethod
    def installITK(confirm=True):
        if confirm and not slicer.app.commandOptions().testingEnabled:
            install = slicer.util.confirmOkCancelDisplay(
                "ITK will be downloaded and installed now. The process might take a minute."
            )
            if not install:
                logging.info("Installation of ITK aborted by the user")
                return None
        slicer.util.pip_install("itk-ants>=0.6.0")
        import itk

        logging.info(f"ITK {itk.__version__} installed correctly")
        return itk


def preloadITK():
    logic = ITKANTsCommonLogic()
    if logic.importITK(True) is not None:
        logic.itk.ANTSRegistration  # trigger loading of itk-ants DLL
