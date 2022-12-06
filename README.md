# 2DCameraContourTracking

## Files
1) Main.py
This contains the main call to the three different subfunctions

2) coordinate_calibration.py
The contained function calibrateCoordinateSystem() can be called to calibrate the coordinate system
It expects 3 dots in a specified distance to each other.

3) calibrate_camera.py
The contained function calibrateCamera() can be called to configure the camera to detect specific parts

4) detect_objects.py
The contained function detectObjects() can be called to track objects in frame and output coordinates in relation to the calibrated coordinate system.

5) helper_functions.py
Containes functions that are needed for other parts of the script
