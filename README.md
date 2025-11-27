**Tag Recognition and Crop — Branch Overview** (Not all files are here, please refer the zip file i sent you on chat) <br> 

This branch contains the trained model and supporting scripts used to detect and crop format-part tags from images or video streams. <br><br>

**📦 What This Branch Contains**<br><br>

A trained YOLO model specifically optimized for detecting format part tags.

Utilities to run the model on individual images, batches, or video frames.

Automatic cropping and saving of each detected tag into a separate output folder for downstream OCR or analysis.<br><br>

**🔄 Workflow**<br>

1. Model Deployment:<br>
The trained model is loaded and deployed into the real-time processing environment.

2. Tag Detection & Cropping:<br>
For each input image or video frame:

All detected format-part tags are cropped.

Each crop is saved into a dedicated output directory.

This enables large-scale extraction of tag regions for further processing.<br><br>

**⚠️ Problem to Address** <br>

In video mode, the same physical tag may appear across multiple consecutive frames.
Current behavior: each frame produces a new crop → many duplicates.<br>
Goal:

Implement a mechanism to avoid re-counting or re-cropping an already-detected tag when it persists across frames.

A potential solution will involve tracking IDs, IoU overlap filtering, or temporal consistency checks.

