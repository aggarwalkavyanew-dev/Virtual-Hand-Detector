# Virtual-Hand-Detector
This Proof of Concept (POC) demonstrates a lightweight, real-time computer vision system designed to detect when a user's hand enters a hazardous area. By creating a virtual boundary on the live camera feed, the system tracks the hand's distance and triggers visual alerts ("DANGER DANGER") upon contact.
Project Name: Real-Time Virtual Hazard Monitor

Overview
A real-time safety monitoring prototype built using Python and OpenCV. The system creates a virtual "Danger Zone" on screen and tracks the user’s hand position relative to it. Unlike standard AI approaches, this project utilizes classical computer vision techniques (HSV color segmentation, Convex Hull geometry, and contour analysis) to achieve high-performance tracking (>30 FPS) on CPU without heavy machine learning dependencies (like MediaPipe). It features a dynamic 3-stage alert system (Safe, Warning, Danger) based on Euclidean distance calculations from the full hand perimeter.

Key Features
Zero-Dependency Tracking: Achieves real-time hand tracking without using MediaPipe, OpenPose, or Neural Networks.
Full-Hand Detection: Uses Convex Hull algorithms to wrap the hand in a geometric polygon, ensuring that knuckles, palms, or wrists trigger the sensor, not just the fingertip.

Dynamic State Logic: Implements a state machine with three distinct zones:
🟢 SAFE: Hand is at a safe distance.
🟡 WARNING: Hand is approaching the object (<150px).
🔴 DANGER: Physical contact detected (Visual Alarm + Red Border).
Optimized Performance: Runs efficiently on standard CPUs using classical image processing techniques.

Technical Stack
Language: Python
Libraries: OpenCV (cv2), NumPy, Math
Techniques: Color Segmentation (HSV), Gaussian Blur, Morphological Operations (Erosion/Dilation), Contour Analysis, Euclidean Distance Calculation.
