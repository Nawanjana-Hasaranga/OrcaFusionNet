# OrcaFusionNet: Bioacoustic Click-Echo Differentiation

This repository contains the Python source code, training pipeline scripts, and architectural diagrams for the paper **"OrcaFusionNet: Fusing High-Definition Spectrograms and Acoustic Metadata for Bioacoustic Click-Echo Differentiation."**

This research was developed within the context of Autonomy Technologies at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), focusing on applied Machine Learning, Computer Vision, and multimodal sensor fusion for bioacoustic environments.

## Project Overview
Manual annotation of acoustic events remains a significant bottleneck in bioacoustics. This project introduces an automated two-stage deep learning pipeline designed to detect and distinguish between clicks and subsequent echoes in odontocete echolocation using underwater recordings of killer whale vocalizations.

### Two-Stage Architecture
1. **Frontline Scanner (YOLOv26):** A YOLO-based object detection pipeline processes composite RGB images (waveform, CWT scalogram, spectrogram) via 5-ms sliding windows to rapidly isolate candidate acoustic events.
2. **Classification (OrcaFusionNet):** A custom Vision Transformer (ViT-B/16) architecture processes composite high-resolution spectrograms fused with a 25-feature acoustic metadata tensor via a custom gated attention mechanism. 

## Performance
Through multi-class threshold calibration applied to weighted focal loss probabilities, the model successfully recovered **2,547 true clicks from 3,752 ground-truth events**, achieving an overall weighted F1-score of **0.5697** in a heavily imbalanced, low-SNR underwater environment.

## Author
* **Nawanjana Hasaranga Fonseka** 
* Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

## Acknowledgments
I would like to express my deepest gratitude to **Christopher Hauer** for giving me the incredible opportunity to work on this project. His extensive mentorship, patience, and continuous support were absolutely instrumental in the success of this research. I also extend a special thanks to **Dr. Heike Vester** for providing the annotated dataset and foundational bioacoustic expertise.