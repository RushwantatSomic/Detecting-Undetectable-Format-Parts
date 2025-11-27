DIGIT UPRIGHT — Branch Overview

This branch contains all scripts and logic used for upright image orientation and numeric tag detection. The workflow ensures that rotated tag images are corrected before running OCR, resulting in significantly improved recognition accuracy.



Image Rotation & Orientation

File: orient_and_fix.py

This script is responsible for automatically correcting the orientation of tag images. It evaluates multiple rotation variants (0°, 90°, 180°, 270°), applies heuristic scoring, and selects the best upright version.
It serves as the preprocessing stage to ensure consistent, rotation-normalized inputs for OCR.



Digit Detection (OCR)

File: ocr_with_rois_v4.py

This script handles the extraction and recognition of digits from the upright tag images. It performs ROI segmentation, per-digit OCR, confidence scoring, and final number assembly.
This module is currently under active development to further improve accuracy and reliability.
