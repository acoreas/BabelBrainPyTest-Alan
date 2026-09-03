import pytest
import os
from glob import glob
import shutil

pytest.mark.generate_outputs
def test_generate_valid_outputs(qtbot,babelbrain_widget,image_to_base64,request,mock_confirm_pseudoct):

    # Save plot screenshot to be added to html report later
    request.node.screenshots = []

    # Start babelbrain instance
    bb_widget = babelbrain_widget(generate_outputs = True)

    outdir = bb_widget.Config['OutputFilesPath']

    finalfile=glob(outdir + os.sep + '*DataForSim-ThermalField-Duration*.h5')
    b_already_calculated=len(finalfile)==1
    if b_already_calculated:
        pytest.skip(f"Calculation already completed previously at {outdir}." )
    if os.path.isdir(outdir): #we just ensure to cleanup to avoid getting stuck in the dialog if we want to reload
        stepsfile=glob(outdir + os.sep + '*.h5')+glob(outdir + os.sep + '*ReuseMask.nii.gz')+\
                  glob(outdir + os.sep + '*BabelViscoInput.nii.gz')+\
                  glob(outdir + os.sep + '*ExecutionTimes.yml')

        for f in stepsfile:
            os.remove(f)
    
    # Run Step 1
    bb_widget.testing_error = False
    bb_widget.Widget.CalculatePlanningMask.click()

    # Wait for step 1 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=2000000)
    qtbot.wait(1000) # Wait for plots to display
    
    # Take screenshot of step 1 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    # Check if step 1 failed
    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 2
    bb_widget.Widget.tabWidget.setCurrentIndex(1)
    #check if Tx with cone design and steering
    if hasattr(bb_widget.AcSim.Widget,'DistanceConeToFocusSpinBox'): 
        DistanceConeToFocusSpinBox=bb_widget.AcSim.Widget.DistanceConeToFocusSpinBox
        minDistance=DistanceConeToFocusSpinBox.minimum()
        maxDistance=DistanceConeToFocusSpinBox.maximum()
        skinToTarget=bb_widget.AcSim.Widget.DistanceSkinLabel.property('UserData')
        ZSteering=0.0
        if skinToTarget >=minDistance and skinToTarget<=maxDistance:
            DistanceConeToFocusSpinBox.setValue(skinToTarget)
        elif skinToTarget>maxDistance:
            DistanceConeToFocusSpinBox.setValue(maxDistance)
            ZSteering=skinToTarget-maxDistance
        else:
            DistanceConeToFocusSpinBox.setValue(minDistance)
            ZSteering=skinToTarget-minDistance
        bb_widget.AcSim.Widget.ZSteeringSpinBox.setValue(ZSteering)

    bb_widget.AcSim.Widget.CalculateAcField.click()

    # Wait for step 2 completion before continuing. Test timeouts after 1hr min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=3600000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 2 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")

    # Run Step 3
    bb_widget.Widget.tabWidget.setCurrentIndex(2)
    bb_widget.ThermalSim.Widget.CalculateThermal.click()

    # Wait for step 3 completion before continuing. Test timeouts after 15 min have past
    qtbot.waitUntil(bb_widget.Widget.tabWidget.isEnabled,timeout=900000)
    qtbot.wait(1000) # Wait for plots to display

    # Take screenshot of step 3 results
    screenshot = qtbot.screenshot(bb_widget)
    request.node.screenshots.append(image_to_base64(screenshot))

    if bb_widget.testing_error == True:
        pytest.fail(f"Test failed due to error in execution")
    
    with qtbot.captureExceptions() as exceptions:
        # Run Export CSV Command
        bb_widget.ThermalSim.Widget.ExportSummary.click()
        
        # Run Export Maps Command
        # bb_widget.ThermalSim.Widget.ExportMaps.click()
        
    if len(exceptions) > 0:
        pytest.fail(f"Test failed due to error in execution: {exceptions}")
    else: #we delete files to reduce space
        L=glob(outdir + os.sep + '*.mat')+\
            glob(outdir + os.sep + '*_Sub.nii.gz')+\
            glob(outdir + os.sep + '*FullElasticSolution.nii.gz')+\
            glob(outdir + os.sep + '*T1W_Resampled.nii.gz')+\
            glob(outdir + os.sep + '*AllCombinations.h5')+\
            glob(outdir + os.sep + '*-PRF-*.nii.gz')+\
            glob(outdir + os.sep + '*DataForSim.h5')+\
            glob(outdir + os.sep + '*Sub_NORM.nii.gz')+\
            glob(outdir + os.sep + 'CT.nii.gz')+\
            glob(outdir + os.sep + 'CT_InT1.nii.gz')+\
            glob(outdir + os.sep + 'm2m_*')+\
            glob(outdir + os.sep + '*FullElasticSolutionPhase.nii.gz')+\
            glob(outdir + os.sep + 'T1*.nii.gz')
        
        for f in L:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)