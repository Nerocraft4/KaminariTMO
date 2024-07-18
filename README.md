# About this repo
The final version of the TMO (Kaminariv3) was released in January 1st, 2024. Minor edits to the algorithm were made after that, to polish the code. 

A file dump/upload was made the 28th of June of 2024, as a backup for the secondary data and tools used in my BSc thesis, presented in February 2024. This includes raw data (psychophisical user inputs and spectroradiometer gray/colour calibration) used when evaluating the algorithm, as well as the plots and documents used in my thesis. An additional backup can be found in this [drive folder](https://drive.google.com/drive/folders/1X_2pT5g8zdCnNkBdzh2xt-lCyHhOw6CD?usp=sharing) (edition dates intact, unlike GitHub), for redundancy purposes.

As of the 18th of July of 2024, another minor edit was made to the GitHub thesis pdf, fixing header/footer issues that went unnoticed, as well as minor edits in the abstract.

# About the source code, KaminariTMO
The code is intended to be used with 12-bit HDR images (supposedly in sRGB/RGB). To use it, simply change the "scene.hdr" by your own filename.

### Pseudocode
```python
Read the image
For each exposition:
  Convert to HSL
  Apply KMeans on Y (lightness) channel
  Apply certain gamma transformation to each pixel, depending on the cluster they are on
Sum & average expositions
Add a touch of saturation to compensate desaturated images
Convert to RGB
Weighted average with the raw image to get dark areas back
Save image as "finalmix.jpg"
```


