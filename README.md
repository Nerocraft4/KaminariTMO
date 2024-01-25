# KaminariTMO
The final version (v3) was released in January 1st, 2024. Minor edits were made after that, to polish the code.

The code is intended to be used with 12-bit HDR images (supposedly in sRGB/RGB). To use it, simply change the "scene.hdr" by your own filename.

### Pseudocode
Read the image

For each exposition:

  Convert to HSL
  
  Apply KMeans on Y (lightness) channel
  
  Apply certain gamma transformation to each pixel, depending on the cluster they are on

Sum and average expositions

Add a touch of saturation to compensate desaturated images

Convert to RGB

Weighted average with the raw image to get dark areas back

Save image as  "finalmix.jpg"
