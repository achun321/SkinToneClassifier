# SkinToneClassifier
Built on [Skin Tone Classifier]([url](https://github.com/ChenglongMa/SkinToneClassifier)https://github.com/ChenglongMa/SkinToneClassifier) by ChelongMa and [SkinDetector]([url](https://github.com/WillBrennan/SkinDetector)https://github.com/WillBrennan/SkinDetector) by WillBrennan.

# Pipeline
1. Detect skin area from image
2. Eliminate harsh shadows / highlights
3. Use k-means clustering to obtain two most dominant colors
4. Check if one of the dominant colors is most a shadow of the actual skin tone color
5. Map to [Monk Scale for Skin Tones]([url](https://skintone.google/)https://skintone.google/)
