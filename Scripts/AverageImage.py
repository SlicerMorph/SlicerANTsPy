import slicer
import ants
import os
import time

# Function to create a Slicer node from an ANTSImage
def nodeFromANTSImage(antsImage, name):
    import ants
    
    imageNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', name)


    tempFilePath = os.path.join(
        slicer.app.temporaryPath,
        "tempImage_{0}.nii.gz".format(time.time()),
    )
    ants.image_write(antsImage, tempFilePath)

    storageNode = slicer.vtkMRMLVolumeArchetypeStorageNode()
    storageNode.SetFileName(tempFilePath)
    storageNode.ReadData(imageNode, True)

    os.remove(tempFilePath)

    return imageNode

# Input directory of aligned images (such as from ANTs Registraion  Groupwise Registration)
input_directory = "D:/ANTsData/MouseOutputs"

# End pattern for selecting the transformed images from the input directory
filename_end_pattern = "transformed.nii.gz"

filePaths = []
for file in os.listdir(input_directory):
    if file.endswith(filename_end_pattern):
        filePaths.append(os.path.join(input_directory, file))


averagedANTSImage = ants.math.average_images(filePaths)

averagedImageNode = nodeFromANTSImage(averagedANTSImage, "AverageImage")

slicer.util.setSliceViewerLayers(background=averagedImageNode, fit=True)



