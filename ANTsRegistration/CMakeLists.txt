#-----------------------------------------------------------------------------
set(MODULE_NAME ANTsPyRegistration)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ANTsPyRegistrationLib/__init__.py
  ANTsPyRegistrationLib/util.py
  ANTsPyRegistrationLib/Widgets/__init__.py
  ANTsPyRegistrationLib/Widgets/delegates.py
  ANTsPyRegistrationLib/Widgets/tables.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Presets/QuickSyN.json
  Resources/Presets/Rigid.json
  Resources/Presets/Rigid+Affine.json
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
