# CNN_BILSTM MODEL FOR OCR PURPOSES

## HOW TO RUN THE MODEL:

1. Install all dependencies
2. Clone the repo
3. Ensure your local repo structure matches the GitHub repo (all files in the same directory)
4. Run the model through terminal with the required parameters:

```bash
python main.py --mode eval --model_path model_assets_final.pth --image_path your_image.jpg --device cuda
```

**Note:** The model works only on line-segmented text. It will hallucinate if the passed image has multiple lines of text or if the text is too small.
