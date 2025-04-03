***READ CAREFULLY***
**CNN_BILSTM MODEL FOR OCR PURPOSES**
*HOW TO RUN THE MODEL:*
First, you should install all dependencies, then clone the repo.
Your local repo after cloning should look the same structure as the github repo(all files are in the same directory).
then you need to run the model thro terminal after passing the parameters as shown in the prompt here:
* python main.py --mode eval --model_path model_assets_final.pth --image_path your_image.jpg --device cuda *
it should work just fine.
*Note: *
the model works only on line segmented text, it will hallucinate if the passed image has multiple lines of text or the text is too small.
