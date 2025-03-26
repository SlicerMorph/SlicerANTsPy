import logging
import os
import json
import glob
import time
from typing import Annotated, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

import vtk, qt, ctk, slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from ITKANTsCommon import ITKANTsCommonLogic

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLTransformNode,
    vtkMRMLLinearTransformNode,
    vtkMRMLBSplineTransformNode,
    vtkMRMLGridTransformNode,
)

from antsRegistrationLib.Widgets.tables import StagesTable, MetricsTable, LevelsTable


ANTsPyTransformTypes  = [ 
    "Rigid",
    "Similarity",
    "Translation",
    "Affine",
    "AffineFast",
    "BOLDAffine",
    "QuickRigid",
    "DenseRigid",
    "BOLDRigid",
    "antsRegistrationSyNQuick[b]",
    "antsRegistrationSyNQuick[s]",
    "antsRegistrationSyNRepro[s]",
    "antsRegistrationSyN[s]",
    "SyNBold",
    "SyNBoldAff",
    "ElasticSyN",
    "SyN",
    "SyNRA",
    "SyNOnly",
    "SyNAggro",
    "SyNCC",
    "TRSAA",
    "SyNabp",
    "SyNLessAggro",
    "TVMSQ",
    "TVMSQC",
]


def ANTsPyTemporaryPath():
    return slicer.util.settingsValue("ANTsPy/TemporaryPath", os.path.join(slicer.app.defaultScenePath, "ANTsPyTemp"))

def writeANTsPyTemporaryPath(path):
    settings = qt.QSettings()

    settings.setValue("ANTsPy/TemporaryPath", path)


def itkTransformFromTransformNode(transformNode):
    """Convert the MRML transform node to an ITK transform."""
    import itk

    if not transformNode:
        return None

    tempFilePath = os.path.join(
        ANTsPyTemporaryPath(),
        "tempTransform_{0}.tfm".format(time.time()),
    )
    storageNode = slicer.vtkMRMLTransformStorageNode()
    storageNode.SetFileName(tempFilePath)
    storageNode.WriteData(transformNode)
    itkTransform = itk.transformread(tempFilePath)
    os.remove(tempFilePath)
    if len(itkTransform) == 1:
        itkTransform = itkTransform[0]
    return itkTransform


def transformNodeFromItkTransform(itkTransform, transformNode=None):
    """Convert the ITK transform to a MRML transform node."""
    import itk

    if not transformNode:
        if isinstance(itkTransform, itk.MatrixOffsetTransformBase):
            transformNode = slicer.vtkMRMLLinearTransformNode()
            slicer.mrmlScene.AddNode(transformNode)
        elif isinstance(itkTransform, itk.BSplineTransform):
            transformNode = slicer.vtkMRMLBSplineTransformNode()
            slicer.mrmlScene.AddNode(transformNode)
        elif isinstance(itkTransform, itk.DisplacementFieldTransform):
            transformNode = slicer.vtkMRMLGridTransformNode()
            slicer.mrmlScene.AddNode(transformNode)
        elif isinstance(itkTransform, itk.CompositeTransform):
            transformNode = slicer.vtkMRMLTransformNode()
            slicer.mrmlScene.AddNode(transformNode)
        else:
            raise ValueError(
                "Unsupported transform type: {0}".format(type(itkTransform))
            )

    tempFilePath = os.path.join(
        ANTsPyTemporaryPath(),
        "tempTransform_{0}.tfm".format(time.time()),
    )
    itk.transformwrite(itkTransform, tempFilePath)
    storageNode = slicer.vtkMRMLTransformStorageNode()
    storageNode.SetFileName(tempFilePath)
    storageNode.ReadData(transformNode, True)
    os.remove(tempFilePath)

    return transformNode

def antsImageFromNode(imageNode):
    import ants
    tempFilePath = os.path.join(
        ANTsPyTemporaryPath(),
        "tempImage_{0}.nii.gz".format(time.time()),
    )

    storageNode = slicer.vtkMRMLVolumeArchetypeStorageNode()
    storageNode.SetFileName(tempFilePath)
    storageNode.WriteData(imageNode)

    image = ants.image_read(tempFilePath)

    os.remove(tempFilePath)

    return image

def nodeFromANTSImage(antsImage, imageNode=None):
    import ants
    
    if not imageNode:
        imageNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')


    tempFilePath = os.path.join(
        ANTsPyTemporaryPath(),
        "tempImage_{0}.nii.gz".format(time.time()),
    )
    ants.image_write(antsImage, tempFilePath)

    storageNode = slicer.vtkMRMLVolumeArchetypeStorageNode()
    storageNode.SetFileName(tempFilePath)
    storageNode.ReadData(imageNode, True)

    os.remove(tempFilePath)

    return imageNode


def nodeFromANTSTransform(antsTransformPath, transformNode):



    storageNode = slicer.vtkMRMLTransformStorageNode()
    storageNode.SetFileName(antsTransformPath)
    storageNode.ReadData(transformNode, True)



def writeTransformSet(outputDirectory, name, direction, transforms):
    
    import shutil
    for i, transform in enumerate(transforms):
        path, ext = os.path.basename(transform).split(os.extsep, 1)
        if ext == 'mat':
            filename = name +'-'+str(i)+ direction  + 'Affine.mat'

        if ext == 'nii.gz':
            filename = name +'-'+str(i) +direction + 'Warp.nii.gz'
        shutil.copy(transform, os.path.join(outputDirectory, filename))


def antsLandmarksFromNode(node):
    pts = []
    for point in range(0, node.GetNumberOfControlPoints()):
        pt = node.GetNthControlPointPosition(point)
        pt = list(pt)
        pt[1] = pt[1] * -1
        pt[0] = pt[0] * -1
        pts.append(pt)

    return np.array(pts)


def createInitialTransform(fixed_landmarks, moving_landmarks, transform_type='rigid', domainImage=None):
    import ants
    fixed_landmarks_ants = antsLandmarksFromNode(fixed_landmarks)
    moving_landmarks_ants = antsLandmarksFromNode(moving_landmarks)
    xfrm = ants.fit_transform_to_paired_points(moving_landmarks_ants, fixed_landmarks_ants)
    tempFilePath = os.path.join(
        ANTsPyTemporaryPath(),
        "tempTransform_{0}.h5".format(time.time()),
    )
    ants.write_transform(xfrm, tempFilePath)

    return [tempFilePath]





class ANTsRegistration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ANTs Registration")
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Registration")
        ]
        self.parent.dependencies = ["ITKANTsCommon"]
        self.parent.contributors = [
            "Dženan Zukić (Kitware Inc.), Simon Oxenford (Netstim Berlin)"
        ]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            "ANTs computes high-dimensional mapping to capture the statistics of brain structure and function."
        )
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Dženan Zukić, Kitware Inc.,
and was partially funded by NIH grant P01HD104435.
"""
        )


class ANTsRegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setEditedNode(self, node, role="", context=""):
        self.setParameterNode(node)
        return node is not None

    def nodeEditable(self, node):
        return (
            0.7
            if node is not None and node.GetAttribute("ModuleName") == self.moduleName
            else 0.0
        )

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        self.uiWidget = slicer.util.loadUI(self.resourcePath("UI/ANTsRegistration.ui"))
        self.layout.addWidget(self.uiWidget)
        self.ui = slicer.util.childWidgetVariables(self.uiWidget)

        self.ui.parameterNodeSelector.addAttribute(
            "vtkMRMLScriptedModuleNode", "ModuleName", self.moduleName
        )
        self.ui.parameterNodeSelector.setNodeTypeLabel(
            "ANTsRegistrationParameters", "vtkMRMLScriptedModuleNode"
        )

        # Set custom UI components

        self.ui.stagesTableWidget = StagesTable()
        stagesTableLayout = qt.QVBoxLayout(self.ui.stagesFrame)
        stagesTableLayout.addWidget(self.ui.stagesTableWidget)

        self.ui.metricsTableWidget = MetricsTable()
        metricsTableLayout = qt.QVBoxLayout(self.ui.metricsFrame)
        metricsTableLayout.addWidget(self.ui.metricsTableWidget)

        self.ui.levelsTableWidget = LevelsTable()
        levelsTableLayout = qt.QVBoxLayout(self.ui.levelsFrame)
        levelsTableLayout.addWidget(self.ui.levelsTableWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        self.uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ANTsRegistrationLogic()

        self.ui.stagesPresetsComboBox.addItems(
            ["Select..."] + PresetManager().getPresetNames()
        )
        self.ui.openPresetsDirectoryButton.clicked.connect(
            self.onOpenPresetsDirectoryButtonClicked
        )

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.parameterNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.setParameterNode
        )
        self.ui.stagesTableWidget.view.selectionModel().selectionChanged.connect(
            self.updateParameterNodeFromGUI
        )
        # self.ui.outputInterpolationComboBox.connect(
        #     "currentIndexChanged(int)", self.updateParameterNodeFromGUI
        # )
        self.ui.outputForwardTransformComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.outputInverseTransformComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.outputVolumeComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.initialTransformTypeComboBox.connect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGUI
        )
        self.ui.initialTransformNodeComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI
        )
        self.ui.dimensionalitySpinBox.connect(
            "valueChanged(int)", self.updateParameterNodeFromGUI
        )
        # self.ui.histogramMatchingCheckBox.connect(
        #     "toggled(bool)", self.updateParameterNodeFromGUI
        # )
        # self.ui.outputDisplacementFieldCheckBox.connect(
        #     "toggled(bool)", self.updateParameterNodeFromGUI
        # )
        # self.ui.winsorizeRangeWidget.connect(
        #     "valuesChanged(double,double)", self.updateParameterNodeFromGUI
        # )
        self.ui.computationPrecisionComboBox.connect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGUI
        )

        self.ui.fixedImageNodeComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateStagesFromFixedMovingNodes
        )
        self.ui.movingImageNodeComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateStagesFromFixedMovingNodes
        )

        # Stages Parameter
        self.ui.stagesTableWidget.removeButton.clicked.connect(
            self.onRemoveStageButtonClicked
        )
        self.ui.metricsTableWidget.removeButton.clicked.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.removeButton.clicked.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.stagesTableWidget.model.itemChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.metricsTableWidget.model.itemChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.model.itemChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.fixedMaskComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateStagesParameterFromGUI
        )
        self.ui.movingMaskComboBox.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.smoothingSigmasUnitComboBox.currentTextChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.convergenceThresholdSpinBox.valueChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.convergenceWindowSizeSpinBox.valueChanged.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.metricsTableWidget.linkStagesPushButton.toggled.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.levelsTableWidget.linkStagesPushButton.toggled.connect(
            self.updateStagesParameterFromGUI
        )
        self.ui.linkMaskingStagesPushButton.toggled.connect(
            self.updateStagesParameterFromGUI
        )

        # Preset Stages
        self.ui.stagesPresetsComboBox.currentTextChanged.connect(self.onPresetSelected)

        # Buttons
        self.ui.stagesTableWidget.savePresetPushButton.connect(
            "clicked(bool)", self.onSavePresetPushButton
        )
        self.ui.runRegistrationButton.connect(
            "clicked(bool)", self.onRunRegistrationButton
        )

        self.ui.fixedImageNodeComboBox.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.movingImageNodeComboBox.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.fixedLandmarkSelector.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.movingLandmarkSelector.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.outputForwardTransformComboBox.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.outputInverseTransformComboBox.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.outputVolumeComboBox.currentNodeChanged.connect(self.checkCanRunPairwiseRegistration)
        self.ui.initialTransformPWCheckBox.toggled.connect(self.checkCanRunPairwiseRegistration)




        self.ui.clearButton.connect("clicked(bool)", self.onClearButton)
        self.ui.removeButton.clicked.connect(self.onRemoveImagePaths)
        self.ui.bumpButton.clicked.connect(self.onBumpImagePath)
        self.ui.selectImages.connect("clicked(bool)", self.onSelectImages)
        self.ui.runTemplateBuilding.connect("clicked(bool)", self.onRunTemplateBuilding)
        self.ui.outTemplateComboBox.currentNodeChanged.connect(self.checkCanRunTemplateBuilding)
        self.ui.outputLandmarksSelector.currentNodeChanged.connect(self.checkCanRunTemplateBuilding)
        self.ui.inTemplateComboBox.currentNodeChanged.connect(self.checkCanRunTemplateBuilding)
        self.ui.inputFileListWidget.currentItemChanged.connect(lambda: qt.QTimer.singleShot(0, self.checkCanRunTemplateBuilding))
        self.ui.initialTransformTBDirectoryButton.directoryChanged.connect(self.checkCanRunTemplateBuilding)
        self.ui.initialTransformTBCheckBox.toggled.connect(self.checkCanRunTemplateBuilding)

        self.ui.inTemplateComboBox.currentNodeChanged.connect(self.checkCanRunGroupRegistration)
        self.ui.inputDirectoryButton.directoryChanged.connect(self.checkCanRunGroupRegistration)
        self.ui.outputDirectoryButton.directoryChanged.connect(self.checkCanRunGroupRegistration)
        self.ui.initialTransformGWCheckBox.toggled.connect(self.checkCanRunGroupRegistration)
        self.ui.templateLandmarksGWSelector.currentNodeChanged.connect(self.checkCanRunGroupRegistration)
        self.ui.initialTransformGWDirectoryButton.directoryChanged.connect(self.checkCanRunGroupRegistration)

        self.ui.jacobianTemplateComboBox.currentNodeChanged.connect(self.checkCanRunAnalysis)
        self.ui.jacobianTemplateComboBox.currentNodeChanged.connect(self.checkCanGenerateImages)
        self.ui.outputImageComboBox.currentNodeChanged.connect(self.checkCanGenerateImages)
        self.ui.jacobianInputListWidget.currentItemChanged.connect(lambda: qt.QTimer.singleShot(0, self.checkCanRunAnalysis))
        self.ui.covariatePathEdit.currentPathChanged.connect(self.checkCanRunAnalysis)
        self.ui.templateMaskComboBox.currentNodeChanged.connect(self.checkCanGenerateImages)
        self.ui.loadPickleButton.clicked.connect(self.unpickleDBM)

        self.ui.runGroupRegistrationButton.clicked.connect(self.runGroupRegistration)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Init tables
        self.ui.stagesTableWidget.view.selectionModel().emitSelectionChanged(
            self.ui.stagesTableWidget.view.selectionModel().selection,
            qt.QItemSelection(),
        )
        self.ui.metricsTableWidget.view.selectionModel().emitSelectionChanged(
            self.ui.metricsTableWidget.view.selectionModel().selection,
            qt.QItemSelection(),
        )

        self.ui.CommonSettings.visible = False
        for transformTypes in ANTsPyTransformTypes:
            self.ui.transformTypeComboBox.addItem(transformTypes)
            self.ui.templateTransformTypeComboBox.addItem(transformTypes)
            self.ui.groupTransformTypeComboBox.addItem(transformTypes)


        self.ui.jacobianInputDirectory.directorySelected.connect(self.populateJacobianInputs)
        self.ui.generateTemplateButton.clicked.connect(self.onGenerateCovariatesTable)
        self.ui.generateJacobianButton.clicked.connect(self.onRunJacobianAnalysis)
        self.ui.generateImageButton.clicked.connect(self.onGenerateImages)

        self.ui.antsPathDirectoryButton.directory = ANTsPyTemporaryPath()

        self.ui.antsPathDirectoryButton.directoryChanged.connect(writeANTsPyTemporaryPath)

        if not os.path.exists(ANTsPyTemporaryPath()):
            os.makedirs(ANTsPyTemporaryPath())
            return

        self.ui.templateOutputDirectoryButton.directory = slicer.app.defaultScenePath


    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()
        self.logic.installANTsPyX()
        self.checkCanRunGroupRegistration()
        self.checkCanRunPairwiseRegistration()
        self.checkCanRunTemplateBuilding()
        self.checkCanRunAnalysis()
        self.checkCanGenerateImages()
        self.setupDBMCache()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(
            self.logic.getParameterNode()
            if not self._parameterNode
            else self._parameterNode
        )

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        wasBlocked = self.ui.parameterNodeSelector.blockSignals(True)
        self.ui.parameterNodeSelector.setCurrentNode(self._parameterNode)
        self.ui.parameterNodeSelector.blockSignals(wasBlocked)

        wasBlocked = self.ui.stagesPresetsComboBox.blockSignals(True)
        self.ui.stagesPresetsComboBox.setCurrentIndex(0)
        self.ui.stagesPresetsComboBox.blockSignals(wasBlocked)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        currentStage = int(
            self._parameterNode.GetParameter(self.logic.params.CURRENT_STAGE_PARAM)
        )
        self.ui.stagesTableWidget.view.setCurrentIndex(
            self.ui.stagesTableWidget.model.index(currentStage, 0)
        )
        self.ui.stagePropertiesCollapsibleButton.text = (
            "Stage " + str(currentStage + 1) + " Properties"
        )
        self.updateStagesGUIFromParameter()

        self.ui.outputForwardTransformComboBox.setCurrentNode(
            self._parameterNode.GetNodeReference(
                self.logic.params.OUTPUT_FORWARD_TRANSFORM_REF
            )
        )
        self.ui.outputInverseTransformComboBox.setCurrentNode(
            self._parameterNode.GetNodeReference(
                self.logic.params.OUTPUT_INVERSE_TRANSFORM_REF
            )
        )
        self.ui.outputVolumeComboBox.setCurrentNode(
            self._parameterNode.GetNodeReference(self.logic.params.OUTPUT_VOLUME_REF)
        )
        # self.ui.outputInterpolationComboBox.currentText = (
        #     self._parameterNode.GetParameter(
        #         self.logic.params.OUTPUT_INTERPOLATION_PARAM
        #     )
        # )
        # self.ui.outputDisplacementFieldCheckBox.checked = int(
        #     self._parameterNode.GetParameter(
        #         self.logic.params.CREATE_DISPLACEMENT_FIELD_PARAM
        #     )
        # )

        self.ui.initialTransformTypeComboBox.currentIndex = (
            int(
                self._parameterNode.GetParameter(
                    self.logic.params.INITIALIZATION_FEATURE_PARAM
                )
            )
            + 2
        )
        self.ui.initialTransformNodeComboBox.setCurrentNode(
            self._parameterNode.GetNodeReference(
                self.logic.params.INITIAL_TRANSFORM_REF
            )
            if self.ui.initialTransformTypeComboBox.currentIndex == 1
            else None
        )
        self.ui.initialTransformNodeComboBox.enabled = (
            self.ui.initialTransformTypeComboBox.currentIndex == 1
        )

        self.ui.dimensionalitySpinBox.value = int(
            self._parameterNode.GetParameter(self.logic.params.DIMENSIONALITY_PARAM)
        )
        # self.ui.histogramMatchingCheckBox.checked = int(
        #     self._parameterNode.GetParameter(self.logic.params.HISTOGRAM_MATCHING_PARAM)
        # )
        # winsorizeIntensities = self._parameterNode.GetParameter(
        #     self.logic.params.WINSORIZE_IMAGE_INTENSITIES_PARAM
        # ).split(",")
        # self.ui.winsorizeRangeWidget.setMinimumValue(float(winsorizeIntensities[0]))
        # self.ui.winsorizeRangeWidget.setMaximumValue(float(winsorizeIntensities[1]))
        # self.ui.computationPrecisionComboBox.currentText = (
        #     self._parameterNode.GetParameter(
        #         self.logic.params.COMPUTATION_PRECISION_PARAM
        #     )
        # )

        
        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateStagesGUIFromParameter(self):
        stagesList = json.loads(
            self._parameterNode.GetParameter(self.logic.params.STAGES_JSON_PARAM)
        )
        self.ui.fixedImageNodeComboBox.setCurrentNodeID(
            stagesList[0]["metrics"][0]["fixed"]
        )
        self.ui.movingImageNodeComboBox.setCurrentNodeID(
            stagesList[0]["metrics"][0]["moving"]
        )
        self.setTransformsGUIFromList(stagesList)
        self.setCurrentStagePropertiesGUIFromList(stagesList)

    def setTransformsGUIFromList(self, stagesList):
        transformsParameters = [stage["transformParameters"] for stage in stagesList]
        self.ui.stagesTableWidget.setGUIFromParameters(transformsParameters)

    def setCurrentStagePropertiesGUIFromList(self, stagesList):
        currentStage = int(
            self._parameterNode.GetParameter(self.logic.params.CURRENT_STAGE_PARAM)
        )
        if {"metrics", "levels", "masks"} <= set(stagesList[currentStage].keys()):
            self.ui.metricsTableWidget.setGUIFromParameters(
                stagesList[currentStage]["metrics"]
            )
            self.ui.levelsTableWidget.setGUIFromParameters(
                stagesList[currentStage]["levels"]
            )
            self.ui.fixedMaskComboBox.setCurrentNodeID(
                stagesList[currentStage]["masks"]["fixed"]
            )
            self.ui.movingMaskComboBox.setCurrentNodeID(
                stagesList[currentStage]["masks"]["moving"]
            )

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = (
            self._parameterNode.StartModify()
        )  # Modify all properties in a single batch

        self._parameterNode.SetParameter(
            self.logic.params.CURRENT_STAGE_PARAM,
            str(self.ui.stagesTableWidget.getSelectedRow()),
        )

        self._parameterNode.SetNodeReferenceID(
            self.logic.params.OUTPUT_FORWARD_TRANSFORM_REF,
            self.ui.outputForwardTransformComboBox.currentNodeID,
        )
        self._parameterNode.SetNodeReferenceID(
            self.logic.params.OUTPUT_INVERSE_TRANSFORM_REF,
            self.ui.outputInverseTransformComboBox.currentNodeID,
        )
        self._parameterNode.SetNodeReferenceID(
            self.logic.params.OUTPUT_VOLUME_REF,
            self.ui.outputVolumeComboBox.currentNodeID,
        )
        # self._parameterNode.SetParameter(
        #     self.logic.params.OUTPUT_INTERPOLATION_PARAM,
        #     self.ui.outputInterpolationComboBox.currentText,
        # )
        # self._parameterNode.SetParameter(
        #     self.logic.params.CREATE_DISPLACEMENT_FIELD_PARAM,
        #     str(int(self.ui.outputDisplacementFieldCheckBox.checked)),
        # )

        self._parameterNode.SetParameter(
            self.logic.params.INITIALIZATION_FEATURE_PARAM,
            str(self.ui.initialTransformTypeComboBox.currentIndex - 2),
        )
        self._parameterNode.SetNodeReferenceID(
            self.logic.params.INITIAL_TRANSFORM_REF,
            self.ui.initialTransformNodeComboBox.currentNodeID,
        )

        self._parameterNode.SetParameter(
            self.logic.params.DIMENSIONALITY_PARAM,
            str(self.ui.dimensionalitySpinBox.value),
        )
        # self._parameterNode.SetParameter(
        #     self.logic.params.HISTOGRAM_MATCHING_PARAM,
        #     str(int(self.ui.histogramMatchingCheckBox.checked)),
        # )
        # self._parameterNode.SetParameter(
        #     self.logic.params.WINSORIZE_IMAGE_INTENSITIES_PARAM,
        #     ",".join(
        #         [
        #             str(self.ui.winsorizeRangeWidget.minimumValue),
        #             str(self.ui.winsorizeRangeWidget.maximumValue),
        #         ]
        #     ),
        # )
        self._parameterNode.SetParameter(
            self.logic.params.COMPUTATION_PRECISION_PARAM,
            self.ui.computationPrecisionComboBox.currentText,
        )

        self._parameterNode.EndModify(wasModified)

    def updateStagesFromFixedMovingNodes(self):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        stagesList = json.loads(
            self._parameterNode.GetParameter(self.logic.params.STAGES_JSON_PARAM)
        )
        for stage in stagesList:
            stage["metrics"][0]["fixed"] = self.ui.fixedImageNodeComboBox.currentNodeID
            stage["metrics"][0][
                "moving"
            ] = self.ui.movingImageNodeComboBox.currentNodeID
        self._parameterNode.SetParameter(
            self.logic.params.STAGES_JSON_PARAM, json.dumps(stagesList)
        )

    def updateStagesParameterFromGUI(self):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        stagesList = json.loads(
            self._parameterNode.GetParameter(self.logic.params.STAGES_JSON_PARAM)
        )
        self.setStagesTransformsToStagesList(stagesList)
        self.setCurrentStagePropertiesToStagesList(stagesList)
        self._parameterNode.SetParameter(
            self.logic.params.STAGES_JSON_PARAM, json.dumps(stagesList)
        )

    def setStagesTransformsToStagesList(self, stagesList):
        for stageNumber, transformParameters in enumerate(
            self.ui.stagesTableWidget.getParametersFromGUI()
        ):
            if stageNumber == len(stagesList):
                stagesList.append({})
            stagesList[stageNumber]["transformParameters"] = transformParameters

    def setCurrentStagePropertiesToStagesList(self, stagesList):
        currentStage = int(
            self._parameterNode.GetParameter(self.logic.params.CURRENT_STAGE_PARAM)
        )

        stagesIterator = (
            range(len(stagesList))
            if self.ui.metricsTableWidget.linkStagesPushButton.checked
            else [currentStage]
        )
        for stageNumber in stagesIterator:
            stagesList[stageNumber][
                "metrics"
            ] = self.ui.metricsTableWidget.getParametersFromGUI()

        stagesIterator = (
            range(len(stagesList))
            if self.ui.levelsTableWidget.linkStagesPushButton.checked
            else [currentStage]
        )
        for stageNumber in stagesIterator:
            stagesList[stageNumber][
                "levels"
            ] = self.ui.levelsTableWidget.getParametersFromGUI()

        stagesIterator = (
            range(len(stagesList))
            if self.ui.linkMaskingStagesPushButton.checked
            else [currentStage]
        )
        for stageNumber in stagesIterator:
            stagesList[stageNumber]["masks"] = {
                "fixed": self.ui.fixedMaskComboBox.currentNodeID,
                "moving": self.ui.movingMaskComboBox.currentNodeID,
            }

    def onRemoveStageButtonClicked(self):
        stagesList = json.loads(
            self._parameterNode.GetParameter(self.logic.params.STAGES_JSON_PARAM)
        )
        if len(stagesList) == 1:
            return
        currentStage = int(
            self._parameterNode.GetParameter(self.logic.params.CURRENT_STAGE_PARAM)
        )
        stagesList.pop(currentStage)
        wasModified = self._parameterNode.StartModify()  # Modify in a single batch
        self._parameterNode.SetParameter(
            self.logic.params.CURRENT_STAGE_PARAM, str(max(currentStage - 1, 0))
        )
        self._parameterNode.SetParameter(
            self.logic.params.STAGES_JSON_PARAM, json.dumps(stagesList)
        )
        self._parameterNode.EndModify(wasModified)

    def onPresetSelected(self, presetName):
        if (
            presetName == "Select..."
            or self._parameterNode is None
            or self._updatingGUIFromParameterNode
        ):
            return
        wasModified = self._parameterNode.StartModify()  # Modify in a single batch
        presetParameters = PresetManager().getPresetParametersByName(presetName)
        for stage in presetParameters["stages"]:
            stage["metrics"][0]["fixed"] = self.ui.fixedImageNodeComboBox.currentNodeID
            stage["metrics"][0][
                "moving"
            ] = self.ui.movingImageNodeComboBox.currentNodeID
        self._parameterNode.SetParameter(
            self.logic.params.STAGES_JSON_PARAM, json.dumps(presetParameters["stages"])
        )
        self._parameterNode.SetParameter(self.logic.params.CURRENT_STAGE_PARAM, "0")
        self._parameterNode.EndModify(wasModified)

    def onSavePresetPushButton(self):
        stages = json.loads(
            self._parameterNode.GetParameter(self.logic.params.STAGES_JSON_PARAM)
        )
        for stage in stages:
            for metric in stage["metrics"]:
                metric["fixed"] = None
                metric["moving"] = None
            stage["masks"]["fixed"] = None
            stage["masks"]["moving"] = None
        savedPresetName = PresetManager().saveStagesAsPreset(stages)
        if savedPresetName:
            self._updatingGUIFromParameterNode = True
            self.ui.stagesPresetsComboBox.addItem(savedPresetName)
            self.ui.stagesPresetsComboBox.setCurrentText(savedPresetName)
            self._updatingGUIFromParameterNode = False

    def onRunRegistrationButton(self):
        # parameters = self.logic.createProcessParameters(self._parameterNode)
        # self.logic.process(**parameters)

        try:
            self.ui.runRegistrationButton.text = "Registration in progress..."
            self.uiWidget.enabled = False
            slicer.app.processEvents()
            with slicer.util.tryWithErrorDisplay("Registration Failed."):
                fixed = self.ui.fixedImageNodeComboBox.currentNode()
                moving = self.ui.movingImageNodeComboBox.currentNode()
                forward = self.ui.outputForwardTransformComboBox.currentNode()
                inverse = self.ui.outputInverseTransformComboBox.currentNode()
                warped = self.ui.outputVolumeComboBox.currentNode()
                fixedLandmarks = self.ui.fixedLandmarkSelector.currentNode()
                movingLandmarks = self.ui.movingLandmarkSelector.currentNode()

                self.logic.process_ANTsPY(
                    self.ui.transformTypeComboBox.currentText,
                   fixed, 
                   moving,
                  forward,
                   inverse, 
                   warped, 
                   self.ui.initialTransformPWCheckBox.checked,
                   fixedLandmarks,
                   movingLandmarks
                   )
            
            self.ui.runRegistrationButton.text = "Run Registration"
            self.uiWidget.enabled = True
            
        except Exception as e:
            self.ui.runRegistrationButton.text = "Run Registration"
            self.uiWidget.enabled = True
           

    def onClearButton(self):
        self.ui.inputFileListWidget.clear()
        self.ui.clearButton.enabled = False

    def onRemoveImagePaths(self):
        selectedItemRows = [self.ui.inputFileListWidget.row(x) for x in self.ui.inputFileListWidget.selectedItems()]

        for i in selectedItemRows:
            self.ui.inputFileListWidget.takeItem(i)

    def onBumpImagePath(self):
        selectedItemRows = [self.ui.inputFileListWidget.row(x) for x in self.ui.inputFileListWidget.selectedItems()]

        for i in selectedItemRows:
            item = self.ui.inputFileListWidget.takeItem(i)
            self.ui.inputFileListWidget.insertItem(0, item)

        self.ui.inputFileListWidget.setCurrentRow(0)

    def onGoToSettings(self):
        self.ui.tabsWidget.setCurrentWidget(self.ui.settingsTab)


    def onSelectImages(self):
        fileDialog = qt.QFileDialog()
        fileDialog.setFileMode(
            qt.QFileDialog.ExistingFiles
        )  # Set to open an existing file
        fileDialog.setNameFilter(
            "Common image types(*.nii.gz *.nrrd *.mha);;All files (*.*)"
        )  # Set file filter
        if fileDialog.exec_():
            inputFilePaths = list(fileDialog.selectedFiles())
        for path in inputFilePaths:
            self.ui.inputFileListWidget.addItem(path)
        self.ui.clearButton.enabled = self.ui.inputFileListWidget.count !=0
        self.checkCanRunTemplateBuilding()

    
    def checkCanRunPairwiseRegistration(self):

        outputSet = self.ui.outputForwardTransformComboBox.currentNode() or self.ui.outputInverseTransformComboBox.currentNode() or self.ui.outputVolumeComboBox.currentNode()

        self.ui.runRegistrationButton.enabled = outputSet and self.ui.fixedImageNodeComboBox.currentNode() and self.ui.movingImageNodeComboBox.currentNode()

        if self.ui.initialTransformPWCheckBox.checked:
            self.ui.runRegistrationButton.enabled = self.ui.runRegistrationButton.enabled and self.ui.fixedLandmarkSelector.currentNode() and self.ui.movingLandmarkSelector.currentNode()
    
    
    def checkCanRunTemplateBuilding(self):

        filePaths = [self.ui.inputFileListWidget.item(x).text() for x in range(self.ui.inputFileListWidget.count)]
        landmarkPaths = self.getInputsFromDirectory(self.ui.initialTransformTBDirectoryButton.directory, ['.mrk.json', '.fcsv'])

        if self.ui.initialTransformTBCheckBox.checked:
            check = self.ui.outTemplateComboBox.currentNode() and len(filePaths) > 0 and (len(landmarkPaths) == len(filePaths)) and self.ui.outputLandmarksSelector.currentNode()
            self.ui.runTemplateBuilding.enabled = check
            if self.ui.initialTemplateComboBox.currentNode():
                 self.ui.runTemplateBuilding.enabled =  self.ui.runTemplateBuilding.enabled and self.ui.templateLandmarksTBSelector.currentNode()

        else:
            self.ui.runTemplateBuilding.enabled = self.ui.outTemplateComboBox.currentNode() and len(filePaths) > 0 
    
    
    def checkCanRunGroupRegistration(self):

        filePaths = self.getInputsFromDirectory(self.ui.inputDirectoryButton.directory, ['.nrrd', '.mha', '.nii.gz'])
        landmarkPaths = self.getInputsFromDirectory(self.ui.initialTransformGWDirectoryButton.directory, ['.mrk.json', '.fcsv'])
        if self.ui.initialTransformGWCheckBox.checked:
            self.ui.runGroupRegistrationButton.enabled = self.ui.inTemplateComboBox.currentNode() and len(filePaths) > 0 and len(landmarkPaths) > 0 and os.path.exists(self.ui.outputDirectoryButton.directory) and self.ui.templateLandmarksGWSelector.currentNode()
        else:
            self.ui.runGroupRegistrationButton.enabled = self.ui.inTemplateComboBox.currentNode() and len(filePaths) > 0 and os.path.exists(self.ui.outputDirectoryButton.directory)


    def checkCanRunAnalysis(self):
        self.ui.generateJacobianButton.enabled = self.ui.jacobianInputListWidget.count !=0 and self.ui.jacobianTemplateComboBox.currentNode() and os.path.exists(self.ui.covariatePathEdit.currentPath)
    
    
    def checkCanGenerateImages(self):
        self.ui.generateImageButton.enabled = self.logic.dbm and self.ui.outputImageComboBox.currentNode() and (self.ui.jacobianTemplateComboBox.currentNode() or self.ui.templateMaskComboBox.currentNode())
    
    
    def onRunTemplateBuilding(self):
        self.uiWidget.enabled = False
        self.ui.runTemplateBuilding.text = "Template building in progess"
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Template building failed."):
                pathList = [self.ui.inputFileListWidget.item(x).text() for x in range(self.ui.inputFileListWidget.count)]
                landmarksPaths = self.getInputsFromDirectory(self.ui.initialTransformTBDirectoryButton.directory, ['.mrk.json', '.fcsv'])
                if self.ui.initialTransformTBCheckBox.checked:
                    self.checkTBLandmarks()
                outputDirectory = os.path.join(self.ui.templateOutputDirectoryButton.directory, "TemplateBuilding_{0}".format(time.time()))
                self.logic.buildTemplateANTsPy(
                    self.ui.initialTemplateComboBox.currentNode(), 
                    pathList, 
                    self.ui.outTemplateComboBox.currentNode(), 
                    self.ui.outputLandmarksSelector.currentNode(),
                    outputDirectory,
                    self.ui.templateTransformTypeComboBox.currentText,
                    self.ui.iterationsSpinBox.value,
                    self.ui.initialTransformTBCheckBox.checked,
                    landmarksPaths,
                    self.ui.templateLandmarksTBSelector.currentNode()

                )
            self.saveTemplateBuildingOutputs(outputDirectory)
            self.uiWidget.enabled = True
            self.ui.runTemplateBuilding.text = "Run Template Building"
            slicer.app.processEvents()
        except Exception as e:
            self.uiWidget.enabled = True
            self.ui.runTemplateBuilding.text = "Run Template Building"
            slicer.app.processEvents()
            raise e
    

    def saveTemplateBuildingOutputs(self, outputDirectory):
        

        templateFilePath = os.path.join(outputDirectory, self.ui.outTemplateComboBox.currentNode().GetName() + ".nii.gz")
        slicer.util.saveNode(self.ui.outTemplateComboBox.currentNode(), templateFilePath)
        if self.ui.outputLandmarksSelector.currentNode():
            landmarksFilePath = os.path.join(outputDirectory, self.ui.outputLandmarksSelector.currentNode().GetName() + ".mrk.json")

            slicer.util.saveNode(self.ui.outputLandmarksSelector.currentNode(), landmarksFilePath)

    
    
    def getInputsFromDirectory(self, directory, extensions):
        filePaths = []
        for file in os.listdir(directory):
            for ext in extensions:
                if file.endswith(ext):
                    filePaths.append(os.path.join(directory, file))
                    break

        return filePaths
    

    def comparePathBasenames(self, primaryList, secondaryList):
        secondaryBasenames = [os.path.basename(x).split(os.extsep, 1)[0] for x in secondaryList]

        for path in primaryList:
            if os.path.basename(path).split(os.extsep, 1)[0] not in secondaryBasenames:
                raise IOError("Missing matching file for: " + path)

    
    def checkGWLandmarks(self):
        imagePaths = self.getInputsFromDirectory(self.ui.inputDirectoryButton.directory, ['.nrrd', '.mha', '.nii.gz'])
        landmarkPaths = self.getInputsFromDirectory(self.ui.initialTransformGWDirectoryButton.directory, ['.mrk.json', '.fcsv'])

        self.comparePathBasenames(imagePaths, landmarkPaths)

    def checkTBLandmarks(self):
        imagePaths = [self.ui.inputFileListWidget.item(x).text() for x in range(self.ui.inputFileListWidget.count)]
        landmarkPaths = self.getInputsFromDirectory(self.ui.initialTransformTBDirectoryButton.directory, ['.mrk.json', '.fcsv'])

        self.comparePathBasenames(imagePaths, landmarkPaths)
    
    def runGroupRegistration(self):
        outputPath  = self.ui.outputDirectoryButton.directory
        filePaths = self.getInputsFromDirectory(self.ui.inputDirectoryButton.directory, ['.nrrd', '.mha', '.nii.gz'])
        landmarksPaths = self.getInputsFromDirectory(self.ui.initialTransformGWDirectoryButton.directory, ['.mrk.json', '.fcsv'])

        self.uiWidget.enabled = False
        self.ui.runGroupRegistrationButton.text = "Group registration in progess"
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Group registration failed."):
                if self.ui.initialTransformGWCheckBox.checked:
                    self.checkGWLandmarks()
                self.logic.groupRegistrationANTsPy(
                    self.ui.inTemplateComboBox.currentNode(), 
                    filePaths, 
                    outputPath, 
                    self.ui.groupTransformTypeComboBox.currentText,
                    self.ui.compositeRadioButton.checked,
                    self.ui.forwardCheckBox.checked,
                    self.ui.inverseCheckBox.checked,
                    self.ui.transformedCheckBox.checked,
                    self.ui.initialTransformGWCheckBox.checked,
                    landmarksPaths,
                    self.ui.templateLandmarksGWSelector.currentNode()

                )
            self.uiWidget.enabled = True
            self.ui.runGroupRegistrationButton.text = "Register"
            slicer.app.processEvents()
        except Exception as e:
            self.uiWidget.enabled = True
            self.ui.runGroupRegistrationButton.text = "Register"
            slicer.app.processEvents()
            raise e

    def populateJacobianInputs(self):
        self.ui.jacobianInputListWidget.clear()
        inputFilePath = self.ui.jacobianInputDirectory.directory

        filePaths = []
        for file in os.listdir(inputFilePath):
            if file.endswith(self.ui.filePatternLineEdit.text):
                filePaths.append(os.path.join(inputFilePath, file))

        for path in filePaths:

            self.ui.jacobianInputListWidget.addItem(os.path.basename(path))


    def updateCovariateFactors(self, covariates):

        self.ui.qValueComboBox.clear()

        for factor in covariates:
            self.ui.qValueComboBox.addItem(factor)
    
    def onGenerateCovariatesTable(self):
        numberOfInputFiles = self.ui.jacobianInputListWidget.count


        files = [os.path.basename(self.ui.jacobianInputListWidget.item(x).text()) for x in range(self.ui.jacobianInputListWidget.count)]

        if numberOfInputFiles<1:
            qt.QMessageBox.critical(slicer.util.mainWindow(),
            'Error', 'Please select input files for analysis before generating covariate table')
            logging.debug('No input files are selected')
            return
        #if #check for rows, columns
        factorList = self.ui.factorLineEdit.text.split(",")
        print(factorList)
        if self.ui.factorLineEdit.text == '' or len(factorList)<1:
            qt.QMessageBox.critical(slicer.util.mainWindow(),
            'Error', 'Please specify at least one factor name to generate a covariate table template')
            logging.debug('No factor names are provided for covariate table template')
            return
        sortedArray = np.zeros(len(files), dtype={'names':('filename', 'procdist'),'formats':('U50','f8')})
        sortedArray['filename']=files

        self.factorTableNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTableNode', 'Factor Table')
        col=self.factorTableNode.AddColumn()
        col.SetName('ID')
        for i in range(len(files)):
            self.factorTableNode.AddEmptyRow()
            self.factorTableNode.SetCellText(i,0,sortedArray['filename'][i])
        for i in range(len(factorList)):
            col=self.factorTableNode.AddColumn()
            col.SetName(factorList[i])
        dateTimeStamp = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        covariateFolder = os.path.join(slicer.app.cachePath, dateTimeStamp)
        self.covariateTableFile = os.path.join(covariateFolder, "covariateTable.csv")
        try:
            os.makedirs(covariateFolder)
            self.covariateTableFile = os.path.join(covariateFolder, "covariateTable.csv")
            slicer.util.saveNode(self.factorTableNode, self.covariateTableFile)
        except:
            logging.debug("Covariate table output failed: Could not write file")
        slicer.mrmlScene.RemoveNode(self.factorTableNode)
        self.ui.covariatePathEdit.currentPath = self.covariateTableFile
        qpath = qt.QUrl.fromLocalFile(os.path.dirname(covariateFolder+os.path.sep))
        qt.QDesktopServices().openUrl(qpath)


    def onRunJacobianAnalysis(self):

        pathList = [self.ui.jacobianInputListWidget.item(x).text() for x in range(self.ui.jacobianInputListWidget.count)]
        pathList = [os.path.join(self.ui.jacobianInputDirectory.directory, x) for x in pathList]
        template = self.ui.jacobianTemplateComboBox.currentNode()
        templateMask = self.ui.templateMaskComboBox.currentNode()
        covariatesFilePath = self.ui.covariatePathEdit.currentPath
        rformula = self.ui.formulaLineEdit.text
        

        self.uiWidget.enabled = False
        self.ui.generateJacobianButton.text = "Jacobian Analysis in progess"
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay("Jacobian Analysis failed."):
                covariates = self.logic.generateJacobian(pathList, template, templateMask,covariatesFilePath, rformula)
            self.updateCovariateFactors(covariates)
            self.uiWidget.enabled = True
            self.ui.generateJacobianButton.text = "Jacobian Analysis"
            self.pickleDBM()
            self.ui.loadCachePathLineEdit.currentPath = self.ui.cachePathLineEdit.currentPath
            self.checkCanGenerateImages()
            slicer.app.processEvents()
        except Exception as e:
            self.uiWidget.enabled = True
            self.ui.generateJacobianButton.text = "Jacobian Analysis"
            slicer.app.processEvents()
            raise e
        

    def onGenerateImages(self):
        template = self.ui.jacobianTemplateComboBox.currentNode()
        templateMask = self.ui.templateMaskComboBox.currentNode()
        outputImage = self.ui.outputImageComboBox.currentNode()
        covariate = self.ui.qValueComboBox.currentText
        
        self.logic.generateImages(covariate, template, templateMask, outputImage)


    def setupDBMCache(self):
        tempFilePath = os.path.join(ANTsPyTemporaryPath(),"dbm.pickle" )
        self.ui.cachePathLineEdit.currentPath = tempFilePath


    def pickleDBM(self):

        import pickle

        with open(self.ui.cachePathLineEdit.currentPath, 'wb') as handle:
            pickle.dump(self.logic.dbm, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def unpickleDBM(self):
        import pickle
        with open(self.ui.loadCachePathLineEdit.currentPath, 'rb') as handle:
            self.logic.dbm = pickle.load(handle)

        covariateOptions = self.logic.dbm['modelNames']
        if 'Intercept' in covariateOptions:
            covariateOptions.remove('Intercept')

        self.updateCovariateFactors(covariateOptions)
        self.checkCanGenerateImages()





    def onOpenPresetsDirectoryButtonClicked(self):
        import platform, subprocess

        presetPath = PresetManager().presetPath
        if platform.system() == "Windows":
            os.startfile(presetPath)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", presetPath])
        else:
            subprocess.Popen(["xdg-open", presetPath])

    


class ANTsRegistrationLogic(ITKANTsCommonLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    @dataclass
    class params:
        OUTPUT_FORWARD_TRANSFORM_REF = "OutputForwardTransform"
        OUTPUT_INVERSE_TRANSFORM_REF = "OutputInverseTransform"
        OUTPUT_VOLUME_REF = "OutputVolume"
        INITIAL_TRANSFORM_REF = "InitialTransform"
        OUTPUT_INTERPOLATION_PARAM = "OutputInterpolation"
        STAGES_JSON_PARAM = "StagesJson"
        CURRENT_STAGE_PARAM = "CurrentStage"
        CREATE_DISPLACEMENT_FIELD_PARAM = "OutputDisplacementField"
        INITIALIZATION_FEATURE_PARAM = "initializationFeature"
        DIMENSIONALITY_PARAM = "Dimensionality"
        HISTOGRAM_MATCHING_PARAM = "HistogramMatching"
        WINSORIZE_IMAGE_INTENSITIES_PARAM = "WinsorizeImageIntensities"
        COMPUTATION_PRECISION_PARAM = "ComputationPrecision"

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ITKANTsCommonLogic.__init__(self)
        if slicer.util.settingsValue(
            "Developer/DeveloperMode", False, converter=slicer.util.toBool
        ):
            import importlib
            import antsRegistrationLib

            antsRegistrationLibPath = os.path.join(
                os.path.dirname(__file__), "antsRegistrationLib"
            )
            G = glob.glob(os.path.join(antsRegistrationLibPath, "**", "*.py"))
            for g in G:
                relativePath = os.path.relpath(
                    g, antsRegistrationLibPath
                )  # relative path
                relativePath = os.path.splitext(relativePath)[0]  # get rid of .py
                moduleParts = relativePath.split(os.path.sep)  # separate
                importlib.import_module(
                    ".".join(["antsRegistrationLib"] + moduleParts)
                )  # import module
                module = antsRegistrationLib
                for (
                    modulePart
                ) in moduleParts:  # iterate over parts in order to load subpkgs
                    module = getattr(module, modulePart)
                importlib.reload(module)  # reload

        self.dbm = None   

    
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        presetParameters = PresetManager().getPresetParametersByName()
        if not parameterNode.GetParameter(self.params.STAGES_JSON_PARAM):
            parameterNode.SetParameter(
                self.params.STAGES_JSON_PARAM, json.dumps(presetParameters["stages"])
            )
        if not parameterNode.GetParameter(self.params.CURRENT_STAGE_PARAM):
            parameterNode.SetParameter(self.params.CURRENT_STAGE_PARAM, "0")

        if not parameterNode.GetNodeReference(self.params.OUTPUT_FORWARD_TRANSFORM_REF):
            parameterNode.SetNodeReferenceID(
                self.params.OUTPUT_FORWARD_TRANSFORM_REF, ""
            )
        if not parameterNode.GetNodeReference(self.params.OUTPUT_INVERSE_TRANSFORM_REF):
            parameterNode.SetNodeReferenceID(
                self.params.OUTPUT_INVERSE_TRANSFORM_REF, ""
            )
        if not parameterNode.GetNodeReference(self.params.OUTPUT_VOLUME_REF):
            parameterNode.SetNodeReferenceID(self.params.OUTPUT_VOLUME_REF, "")
        if not parameterNode.GetParameter(self.params.OUTPUT_INTERPOLATION_PARAM):
            parameterNode.SetParameter(
                self.params.OUTPUT_INTERPOLATION_PARAM,
                str(presetParameters["outputSettings"]["interpolation"]),
            )
        if not parameterNode.GetParameter(self.params.CREATE_DISPLACEMENT_FIELD_PARAM):
            parameterNode.SetParameter(self.params.CREATE_DISPLACEMENT_FIELD_PARAM, "0")

        if not parameterNode.GetParameter(self.params.INITIALIZATION_FEATURE_PARAM):
            parameterNode.SetParameter(
                self.params.INITIALIZATION_FEATURE_PARAM,
                str(
                    presetParameters["initialTransformSettings"][
                        "initializationFeature"
                    ]
                ),
            )
        if not parameterNode.GetNodeReference(self.params.INITIAL_TRANSFORM_REF):
            parameterNode.SetNodeReferenceID(self.params.INITIAL_TRANSFORM_REF, "")

        if not parameterNode.GetParameter(self.params.DIMENSIONALITY_PARAM):
            parameterNode.SetParameter(
                self.params.DIMENSIONALITY_PARAM,
                str(presetParameters["generalSettings"]["dimensionality"]),
            )
        if not parameterNode.GetParameter(self.params.HISTOGRAM_MATCHING_PARAM):
            parameterNode.SetParameter(
                self.params.HISTOGRAM_MATCHING_PARAM,
                str(presetParameters["generalSettings"]["histogramMatching"]),
            )
        if not parameterNode.GetParameter(
            self.params.WINSORIZE_IMAGE_INTENSITIES_PARAM
        ):
            parameterNode.SetParameter(
                self.params.WINSORIZE_IMAGE_INTENSITIES_PARAM,
                ",".join(
                    [
                        str(x)
                        for x in presetParameters["generalSettings"][
                            "winsorizeImageIntensities"
                        ]
                    ]
                ),
            )
        if not parameterNode.GetParameter(self.params.COMPUTATION_PRECISION_PARAM):
            parameterNode.SetParameter(
                self.params.COMPUTATION_PRECISION_PARAM,
                presetParameters["generalSettings"]["computationPrecision"],
            )

    def createProcessParameters(self, paramNode):
        parameters = {}
        parameters["stages"] = json.loads(
            paramNode.GetParameter(self.params.STAGES_JSON_PARAM)
        )

        # ID to Node
        for stage in parameters["stages"]:
            for metric in stage["metrics"]:
                metric["fixed"] = (
                    slicer.util.getNode(metric["fixed"]) if metric["fixed"] else ""
                )
                metric["moving"] = (
                    slicer.util.getNode(metric["moving"]) if metric["moving"] else ""
                )
            stage["masks"]["fixed"] = (
                slicer.util.getNode(stage["masks"]["fixed"])
                if stage["masks"]["fixed"]
                else ""
            )
            stage["masks"]["moving"] = (
                slicer.util.getNode(stage["masks"]["moving"])
                if stage["masks"]["moving"]
                else ""
            )

        parameters["outputSettings"] = {}
        parameters["outputSettings"]["forwardTransform"] = paramNode.GetNodeReference(
            self.params.OUTPUT_FORWARD_TRANSFORM_REF
        )
        parameters["outputSettings"]["inverseTransform"] = paramNode.GetNodeReference(
            self.params.OUTPUT_INVERSE_TRANSFORM_REF
        )
        parameters["outputSettings"]["volume"] = paramNode.GetNodeReference(
            self.params.OUTPUT_VOLUME_REF
        )
        parameters["outputSettings"]["interpolation"] = paramNode.GetParameter(
            self.params.OUTPUT_INTERPOLATION_PARAM
        )
        parameters["outputSettings"]["useDisplacementField"] = int(
            paramNode.GetParameter(self.params.CREATE_DISPLACEMENT_FIELD_PARAM)
        )

        parameters["initialTransformSettings"] = {}
        parameters["initialTransformSettings"]["initializationFeature"] = int(
            paramNode.GetParameter(self.params.INITIALIZATION_FEATURE_PARAM)
        )
        parameters["initialTransformSettings"]["initialTransformNode"] = (
            paramNode.GetNodeReference(self.params.INITIAL_TRANSFORM_REF)
        )

        parameters["generalSettings"] = {}
        parameters["generalSettings"]["dimensionality"] = int(
            paramNode.GetParameter(self.params.DIMENSIONALITY_PARAM)
        )
        parameters["generalSettings"]["histogramMatching"] = int(
            paramNode.GetParameter(self.params.HISTOGRAM_MATCHING_PARAM)
        )
        parameters["generalSettings"]["winsorizeImageIntensities"] = [
            float(val)
            for val in paramNode.GetParameter(
                self.params.WINSORIZE_IMAGE_INTENSITIES_PARAM
            ).split(",")
        ]
        parameters["generalSettings"]["computationPrecision"] = paramNode.GetParameter(
            self.params.COMPUTATION_PRECISION_PARAM
        )

        return parameters

    
    
    def process_ANTsPY(
            self,
            transformType,
            fixedNode,
            movingNode,
            forwardTransformNode,
            inverseTransformNode,
            transformedImageNode,
            useLandmarks,
            fixedLandmarks,
            movingLandmarks,

    ):
        import ants

        fixedImage = antsImageFromNode(fixedNode)
        movingImage = antsImageFromNode(movingNode)
        initialTransform = None

        if useLandmarks:
            initialTransform = createInitialTransform(fixedLandmarks, movingLandmarks)

        reg = ants.registration(fixed=fixedImage, moving=movingImage, type_of_transform=transformType, write_composite_transform=True, initial_transform=initialTransform)

        if forwardTransformNode:
            nodeFromANTSTransform(reg['fwdtransforms'], forwardTransformNode)

        if inverseTransformNode:
            nodeFromANTSTransform(reg['invtransforms'], inverseTransformNode)

        if transformedImageNode:
            nodeFromANTSImage(reg['warpedmovout'], transformedImageNode)
            slicer.util.setSliceViewerLayers(background = transformedImageNode)
    
    
    
    def process(
        self,
        stages,
        outputSettings,
        initialTransformSettings=None,
        generalSettings=None,
        wait_for_completion=False,
    ):
        """
        :param stages: list defining registration stages
        :param outputSettings: dictionary defining output settings
        :param initialTransformSettings: dictionary defining initial moving transform
        :param generalSettings: dictionary defining general registration settings
        :param wait_for_completion: flag to enable waiting for completion
        See presets examples to see how these are specified
        """

        if generalSettings is None:
            generalSettings = {}
        if initialTransformSettings is None:
            initialTransformSettings = {}

        logging.info("Instantiating the filter")
        slicer.app.processEvents()
        itk = self.itk
        precision_type = itk.F
        if generalSettings["computationPrecision"] == "double":
            precision_type = itk.D
        fixedImage = slicer.util.itkImageFromVolume(stages[0]["metrics"][0]["fixed"])
        movingImage = slicer.util.itkImageFromVolume(stages[0]["metrics"][0]["moving"])

        initial_itk_transform = itk.AffineTransform[
            precision_type, fixedImage.ndim
        ].New()
        initial_itk_transform.SetIdentity()
        if "initialTransformNode" in initialTransformSettings:
            if initialTransformSettings["initialTransformNode"]:
                initial_itk_transform = itkTransformFromTransformNode(
                    initialTransformSettings["initialTransformNode"]
                )
        elif "initializationFeature" in initialTransformSettings:
            print("This initialization is not yet implemented")
            # use itk.CenteredTransformInitializer to construct initial transform

        slicer.app.processEvents()
        startTime = time.time()
        ants_reg = itk.ANTSRegistration[
            type(fixedImage), type(movingImage), precision_type
        ].New()

        
        for stage_index, stage in enumerate(stages):
            ants_reg.SetFixedImage(fixedImage)
            ants_reg.SetMovingImage(movingImage)
            ants_reg.SetInitialTransform(initial_itk_transform)
            assert fixedImage.ndim == movingImage.ndim
            assert fixedImage.ndim == generalSettings["dimensionality"]
            # currently unexposed parameters
            # generalSettings["winsorizeImageIntensities"]
            # generalSettings["histogramMatching"]
            # outputSettings["interpolation"]
            # outputSettings["useDisplacementField"]
            logging.info(f"Stage {stage_index} started")

            transform_type = stage["transformParameters"]["transform"]
            if transform_type == "SyN":
                transform_type = "SyNOnly"
            ants_reg.SetTypeOfTransform(transform_type)
            transform_settings = stage["transformParameters"]["settings"].split(",")
            ants_reg.SetGradientStep(float(transform_settings[0]))
            # TODO: other parameters depend on the type of transform, see
            # https://github.com/ANTsX/ANTs/blob/beb4aa2e9456445249de6ae6698e3f6ed8c4767b/Examples/antsRegistration.cxx#L370-L391

            assert len(stage["metrics"]) == 1
            metric_type = stage["metrics"][0]["type"]

            metric_settings = stage["metrics"][0]["settings"].split(",")
            if metric_type in ["MI", "Mattes"]:
                ants_reg.SetNumberOfBins(int(metric_settings[1]))
            else:
                ants_reg.SetRadius(int(metric_settings[1]))
            # if len(metric_settings) > 2:
            #     ants_reg.SetSamplingStrategy(metric_settings[2])  # not exposed
            if len(metric_settings) > 3:
                ants_reg.SetSamplingRate(float(metric_settings[3]))
            if len(metric_settings) > 4:
                ants_reg.SetUseGradientFilter(bool(metric_settings[4]))
            iterations = []
            shrink_factors = []
            sigmas = []
            for step in stage["levels"]["steps"]:
                iterations.append(step["convergence"])
                shrink_factors.append(step["shrinkFactors"])
                sigmas.append(step["smoothingSigmas"])
            ants_reg.SetShrinkFactors(shrink_factors)
            ants_reg.SetSmoothingSigmas(sigmas)
            ants_reg.SetSmoothingInPhysicalUnits(
                stage["levels"]["smoothingSigmasUnit"] == "mm"
            )
            # not exposed:
            # stage["levels"]["smoothingSigmasUnit"]
            # stage["levels"]["convergenceThreshold"]
            if transform_type in [
                "Rigid",
                "Affine",
                "CompositeAffine",
                "Similarity",
                "Translation",
            ]:
                ants_reg.SetAffineMetric(metric_type)
                ants_reg.SetAffineIterations(iterations)
            else:
                ants_reg.SetSynMetric(metric_type)
                ants_reg.SetSynIterations(iterations)
            if stage["masks"]["fixed"] is not None and stage["masks"]["fixed"] != "":
                ants_reg.SetFixedImageMask(
                    slicer.util.itkImageFromVolume(stage["masks"]["fixed"])
                )
            if stage["masks"]["moving"] is not None and stage["masks"]["moving"] != "":
                ants_reg.SetMovingImageMask(
                    slicer.util.itkImageFromVolume(stage["masks"]["moving"])
                )
            ants_reg.Update()
            initial_itk_transform = ants_reg.GetForwardTransform()
            slicer.app.processEvents()
            # TODO: update progress bar
        forwardTransform = ants_reg.GetForwardTransform()
        inverseTransform = ants_reg.GetInverseTransform()
        slicer.app.processEvents()
        if outputSettings["forwardTransform"] is not None:
            transformNodeFromItkTransform(
                forwardTransform, outputSettings["forwardTransform"]
            )
        if outputSettings["inverseTransform"] is not None:
            transformNodeFromItkTransform(
                inverseTransform, outputSettings["inverseTransform"]
            )

        if outputSettings["volume"] is not None:
            itkImage = ants_reg.GetWarpedMovingImage()
            slicer.util.updateVolumeFromITKImage(outputSettings["volume"], itkImage)
            slicer.util.setSliceViewerLayers(
                background=outputSettings["volume"], fit=True, rotateToVolumePlane=True
            )

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    
    def getLandmarksForImage(self, imagePath, landmarkPaths):
        basename = os.path.basename(imagePath).split(os.extsep, 1)[0]

        for p in landmarkPaths:
            if basename in p:
                return p
    
    
    def groupRegistrationANTsPy(
            self, 
            template, 
            pathlist, 
            outputDirectory, 
            transformtype, 
            writeCompositeTransform=False, 
            outputForward=True, 
            outputInverse=True, 
            outputTransformed=True,
            useLandmarks=False,
            landmarksPaths=None,
            templateLandmarks = None
            ):
        
        import ants
        import shutil
        fixed = antsImageFromNode(template)

        for i, path in enumerate(pathlist):

            print("Registering image {0} of {1}".format(i+1, len(pathlist)))
            slicer.app.processEvents()
            initialTransform = None
            if useLandmarks:
                imageLandmarksPath = self.getLandmarksForImage(path, landmarksPaths)
                imageLandmarks = slicer.util.loadMarkups(imageLandmarksPath)
                initialTransform = createInitialTransform(templateLandmarks, imageLandmarks)
                slicer.mrmlScene.RemoveNode(imageLandmarks)
            name, ext = os.path.basename(path).split(os.extsep, 1)
            moving = ants.image_read(path)
            reg = ants.registration(fixed=fixed, moving=moving, initial_transform=initialTransform,type_of_transform=transformtype, write_composite_transform=writeCompositeTransform)
            if writeCompositeTransform:
                if outputForward:
                    forwardName = os.path.join(outputDirectory, name +'-forward.h5')
                    shutil.copy(reg['fwdtransforms'], forwardName)
                if outputInverse:
                    inverseName = os.path.join(outputDirectory, name +'-inverse.h5')
                    shutil.copy(reg['invtransforms'], inverseName)
            else:
                if outputForward:
                    writeTransformSet(outputDirectory, name, 'forward', reg['fwdtransforms'])
                if outputInverse:
                    writeTransformSet(outputDirectory, name, 'inverse', reg['invtransforms'])
            if outputTransformed:
                transformedName = os.path.join(outputDirectory, name +'-transformed.nii.gz')
                ants.image_write(reg['warpedmovout'], transformedName)
            
    
    
    def getAlignedImages(self,
                        pathList,
                        landmarksPaths,
                        initialTemplate,
                        templateLandmarks):

        import ants

        if initialTemplate:
            fixedImage = antsImageFromNode(initialTemplate)
            fixedLandmarks = templateLandmarks
        else:
            fixedImage = ants.image_read(pathList[0])
            fixedLandmarksPath = self.getLandmarksForImage(pathList[0], landmarksPaths)
            fixedLandmarks = slicer.util.loadMarkups(fixedLandmarksPath)

        imageList = []

        for path in pathList:
            movingImage = ants.image_read(path)
            landmarkPath = self.getLandmarksForImage(path, landmarksPaths)
            movingLandmarks = slicer.util.loadMarkups(landmarkPath)

            initialTransform = createInitialTransform(fixedLandmarks, movingLandmarks)

            alignedImage = ants.apply_transforms(fixedImage, movingImage, initialTransform)

            imageList.append(alignedImage)

            slicer.mrmlScene.RemoveNode(movingLandmarks)

        if not initialTemplate:
            slicer.mrmlScene.RemoveNode(fixedLandmarks)

        return imageList

        
    
    
    def buildTemplateANTsPy(self, 
                            initialTemplate, 
                            pathList, 
                            outputTemplate,
                            outputLandmarks,
                            outputDirectory,
                            transformType, 
                            iterations=1, 
                            useLandmarks=False, 
                            landmarksPaths=None, 
                            templateLandmarks=None):
        import ants
        antsInitialTemplate = None
        if initialTemplate:
            antsInitialTemplate = antsImageFromNode(initialTemplate)

        if useLandmarks:
            imageList = self.getAlignedImages(pathList, landmarksPaths, initialTemplate, templateLandmarks)
        else:
            imageList = []
            for path in pathList:
                antsImage = ants.image_read(path)
                imageList.append(antsImage)

        template = antsInitialTemplate       

        for i in range(0, iterations):
            print("Template building iteration {0}".format(i))
            slicer.app.processEvents()
            template = ants.build_template(initial_template=template, image_list=imageList, type_of_transform=transformType, iterations=1)

            # If we are not on the last iteration, save the intermediate
            if i < iterations - 1:
                templateFilePath = os.path.join(outputDirectory, outputTemplate.GetName() + "-" + str(i) +".nii.gz")
                if not os.path.exists(outputDirectory):
                    os.makedirs(outputDirectory)
                ants.image_write(template, templateFilePath)

        nodeFromANTSImage(template, outputTemplate)

        slicer.util.setSliceViewerLayers(background=outputTemplate, fit=True)

        if useLandmarks:
            if initialTemplate:
                self.copyLandmarks(templateLandmarks, outputLandmarks)
            else:
                fixedLandmarksPath = self.getLandmarksForImage(pathList[0], landmarksPaths)
                firstLandmarks = slicer.util.loadMarkups(fixedLandmarksPath)
                self.copyLandmarks(firstLandmarks, outputLandmarks)
                slicer.mrmlScene.RemoveNode(firstLandmarks)

            
    
    def copyLandmarks(self, source, destination, clear=True):
        if clear:
            destination.RemoveAllControlPoints()

        for point in range(0, source.GetNumberOfControlPoints()):
            pt = source.GetNthControlPointPosition(point)
            destination.AddControlPoint(pt)

    
    def buildTemplate(
        self, initialTemplate, pathList, outputTemplate, stages, generalSettings
    ):
        if len(stages) > 1:
            logging.error(
                "Template building is not yet implemented for multiple stages"
            )
            return

        logging.info("Preparing to build the template image")
        itk = self.itk
        precision_type = itk.F
        if generalSettings["computationPrecision"] == "double":
            precision_type = itk.D

        # read first image - it will be used as a reference for image grid, pixel type, and dimensionality
        itk = self.itk
        firstImage = itk.imread(pathList[0])

        template_type = type(firstImage)
        if initialTemplate is not None:
            initialTemplateImage = slicer.util.itkImageFromVolume(initialTemplate)
            template_type = type(initialTemplateImage)

        gwtb = itk.ANTSGroupwiseBuildTemplate[type(firstImage), template_type, precision_type].New()
        if initialTemplate is not None:
            gwtb.SetInitialTemplateImage(initialTemplateImage)
        gwtb.SetPathList(pathList)

        stage = stages[0]
        # TODO: construct pairwise registration instance and set it to gwtb
        ants_reg = itk.ANTSRegistration[
            type(firstImage), template_type, precision_type
        ].New()

        transform_type = stage["transformParameters"]["transform"]
        if transform_type == "SyN":
            transform_type = "SyNOnly"
        ants_reg.SetTypeOfTransform(transform_type)
        transform_settings = stage["transformParameters"]["settings"].split(",")
        ants_reg.SetGradientStep(float(transform_settings[0]))

        

        assert len(stage["metrics"]) == 1
        metric_type = stage["metrics"][0]["type"]

        metric_settings = stage["metrics"][0]["settings"].split(",")
        if metric_type in ["MI", "Mattes"]:
            ants_reg.SetNumberOfBins(int(metric_settings[1]))
        else:
            ants_reg.SetRadius(int(metric_settings[1]))
        # if len(metric_settings) > 2:
        #     ants_reg.SetSamplingStrategy(metric_settings[2])  # not exposed
        if len(metric_settings) > 3:
            ants_reg.SetSamplingRate(float(metric_settings[3]))
        if len(metric_settings) > 4:
            ants_reg.SetUseGradientFilter(bool(metric_settings[4]))
        iterations = []
        shrink_factors = []
        sigmas = []
        for step in stage["levels"]["steps"]:
            iterations.append(step["convergence"])
            shrink_factors.append(step["shrinkFactors"])
            sigmas.append(step["smoothingSigmas"])
        ants_reg.SetShrinkFactors(shrink_factors)
        ants_reg.SetSmoothingSigmas(sigmas)
        ants_reg.SetSmoothingInPhysicalUnits(
            stage["levels"]["smoothingSigmasUnit"] == "mm"
        )
        # not exposed:
        # stage["levels"]["smoothingSigmasUnit"]
        # stage["levels"]["convergenceThreshold"]
        if transform_type in [
            "Rigid",
            "Affine",
            "CompositeAffine",
            "Similarity",
            "Translation",
        ]:
            ants_reg.SetAffineMetric(metric_type)
            ants_reg.SetAffineIterations(iterations)
        else:
            ants_reg.SetSynMetric(metric_type)
            ants_reg.SetSynIterations(iterations)

        gwtb.SetPairwiseRegistration(ants_reg)

        logging.info("Running ANTSGroupwiseBuildTemplate")
        slicer.app.processEvents()
        startTime = time.time()
        gwtb.Update()
        stopTime = time.time()
        logging.info(f"ANTSGroupwiseBuildTemplate completed in {stopTime-startTime:.2f} seconds")

        if outputTemplate is not None:
            itkImage = gwtb.GetOutput()
            slicer.util.updateVolumeFromITKImage(outputTemplate, itkImage)
            slicer.util.setSliceViewerLayers(
                background=outputTemplate, fit=True, rotateToVolumePlane=True
            )

    def installANTsPyX(self):

        try:
            import ants
        except:
            slicer.util.pip_install('antspyx')


    def generateJacobian(self, pathList, templateNode, templateMaskNode, covariatesFilePath, rformula):

        import ants

        template = antsImageFromNode(templateNode)
        if templateMaskNode:
            raw_mask = antsImageFromNode(templateMaskNode)
            template_mask = ants.get_mask(raw_mask,1, 100, 0)
        else:

            template_mask = ants.get_mask(template)

        
        log_jacobian_image_list = list()

        for path in pathList:
            jacobian = ants.create_jacobian_determinant_image(template, path, do_log=True)
            log_jacobian_image_list.append(jacobian)

        log_jacobian = ants.image_list_to_matrix(log_jacobian_image_list, template_mask)

        import pandas

        df = pandas.read_csv(covariatesFilePath)
        availableFactors = df.columns.to_list()
        availableFactors.remove('ID')

        print("Factors from csv file: ")
        print(availableFactors)

        data = {}

        for factor in availableFactors:
            factorValues = df[factor].to_numpy()
            data[factor] = factorValues

        covariates = pandas.DataFrame(data)

        print(covariates)
        print(rformula)


        self.dbm = ants.ilr(covariates, {"log_jacobian" : log_jacobian}, rformula, verbose=True)


        covariateOptions = self.dbm['modelNames']
        if 'Intercept' in covariateOptions:
            covariateOptions.remove('Intercept')

        return covariateOptions



    def generateImages(self, targetCovariate, templateNode, templateMaskNode,outputImageNode):
        import statsmodels
        import ants

        if templateMaskNode:
            raw_mask = antsImageFromNode(templateMaskNode)
            template_mask = ants.get_mask(raw_mask,1, 100, 0)
        else:
            template = antsImageFromNode(templateNode)
            template_mask = ants.get_mask(template)

        log_jacobian_p_values = self.dbm['pValues']['pval_'+ targetCovariate]
        log_jacobian_q_values = statsmodels.stats.multitest.fdrcorrection(log_jacobian_p_values, alpha=0.05, method='poscorr', is_sorted=False)[1]
        log_jacobian_beta_values = self.dbm['coefficientValues']['coef_'+ targetCovariate]
        log_jacobian_q_values_image = ants.matrix_to_images(np.reshape(log_jacobian_q_values, (1, len(log_jacobian_q_values))), template_mask)[0]
        log_jacobian_beta_values_image = ants.matrix_to_images(np.reshape(log_jacobian_beta_values, (1, len(log_jacobian_beta_values))), template_mask)[0]
        output_Image = ants.mask_image(log_jacobian_beta_values_image, ants.get_mask(log_jacobian_q_values_image))


        # get index of the correct covariate

        nodeFromANTSImage(output_Image, outputImageNode)



            
            


class PresetManager:
    def __init__(self):
        self.presetPath = os.path.join(
            os.path.dirname(__file__), "Resources", "Presets"
        )

    def saveStagesAsPreset(self, stages):
        from PythonQt import BoolResult

        ok = BoolResult()
        presetName = qt.QInputDialog().getText(
            qt.QWidget(),
            "Save Preset",
            "Preset name: ",
            qt.QLineEdit.Normal,
            "my_preset",
            ok,
        )
        if not ok:
            return
        if presetName in self.getPresetNames():
            slicer.util.warningDisplay(
                f"{presetName} already exists. Set another name."
            )
            return self.saveStagesAsPreset(stages)
        outFilePath = os.path.join(self.presetPath, f"{presetName}.json")
        saveSettings = self.getPresetParametersByName()
        saveSettings["stages"] = stages
        try:
            with open(outFilePath, "w") as outfile:
                json.dump(saveSettings, outfile)
        except:
            slicer.util.warningDisplay(f"Unable to write into {outFilePath}")
            return
        slicer.util.infoDisplay(f"Saved preset to {outFilePath}.")
        return presetName

    def getPresetParametersByName(self, name="Rigid"):
        presetFilePath = os.path.join(self.presetPath, name + ".json")
        with open(presetFilePath) as presetFile:
            return json.load(presetFile)

    def getPresetNames(self):
        G = glob.glob(os.path.join(self.presetPath, "*.json"))
        return [os.path.splitext(os.path.basename(g))[0] for g in G]


class ANTsRegistrationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ANTsRegistration1()

    def test_ANTsRegistration1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        sampleDataLogic = SampleData.SampleDataLogic()
        fixed = sampleDataLogic.downloadMRBrainTumor1()
        moving = sampleDataLogic.downloadMRBrainTumor2()

        initialTransform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode"
        )
        outputForwardTransform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTransformNode"
        )
        outputInverseTransform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTransformNode"
        )
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

        logic = ANTsRegistrationLogic()
        presetParameters = PresetManager().getPresetParametersByName("QuickSyN")
        for stage in presetParameters["stages"]:
            for metric in stage["metrics"]:
                metric["fixed"] = fixed
                metric["moving"] = moving
            # let's make it quick
            for step in stage["levels"]["steps"]:
                step["shrinkFactors"] = 10
            stage["levels"]["convergenceThreshold"] = 1
            stage["levels"]["convergenceWindowSize"] = 5

        presetParameters["initialTransformSettings"][
            "initialTransformNode"
        ] = initialTransform
        presetParameters["outputSettings"]["forwardTransform"] = outputForwardTransform
        presetParameters["outputSettings"]["inverseTransform"] = outputInverseTransform
        presetParameters["outputSettings"]["volume"] = outputVolume
        presetParameters["outputSettings"]["log"] = None

        logic.process(**presetParameters)

        self.delayDisplay("Test passed!")
