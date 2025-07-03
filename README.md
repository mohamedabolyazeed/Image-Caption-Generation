# Image Caption Generator: Turning Pixels into Poetry!

Welcome to the magical world of **Image Caption Generator**, where silent pixels come alive with stories! Using the power of Deep Learning, we’ve crafted a system that looks at an image and spins a tale—like a poet gazing at a canvas. Powered by **DenseNet201 (CNN)** for vision and **LSTM (RNN)** for storytelling, this project is a blend of tech and creativity. Let’s dive into this enchanting journey! 

## Project Overview: What’s the Magic About?

Imagine uploading a photo of a child climbing a playhouse, and voila! The system whispers: *"A little girl in a pink dress climbing into a wooden playhouse."* That’s the magic we’re crafting here! This project uses AI to automatically generate captions for images, making the world more accessible, searchable, and poetic.

### Why This Matters?
- **Accessibility**: Helps visually impaired folks “hear” images through words.
- **Social Media Magic**: Auto-generates captions for your Instagram posts! 📸
- **Search Power**: Makes images searchable by linking them to text.

## The Spellbook: Tech and Tools We Used

To cast this spell, we gathered some powerful ingredients:
- **Dataset**: The **Flickr8k** dataset—a treasure chest of 8,000 images with 5 captions each, perfect for training our AI poet.
- **CNN (DenseNet201)**: The “vision master” that scans images and extracts features like shapes, colors, and objects.
- **RNN (LSTM)**: The “story weaver” that takes those features and crafts captions word by word.
- **Libraries**:
  - `TensorFlow` and `Keras`: For building and training our models.
  - `Pandas` and `NumPy`: For handling data like a pro.
  - `Matplotlib` and `Seaborn`: For visualizing results with flair.
  - `Tqdm`: To keep track of progress without losing our minds!

## How the Magic Happens: Step-by-Step

Here’s how we brewed this potion in the Jupyter Notebook:

### 1. **Setting the Stage: Install and Prep**
We kicked things off by setting up our environment on Google Colab:
- Installed the `kaggle` library to fetch our dataset.
- Uploaded our `kaggle.json` credentials (don’t worry, we kept it secret!).
- Code snippet:
  ```
  !pip install kaggle
  from google.colab import files
  files.upload()
  ```
- Created a cozy home for our credentials and downloaded the **Flickr8k** dataset:
  ```
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json
  !kaggle datasets download -d adityajn105/flickr8k
  !unzip flickr8k.zip -d flickr8k
  ```

### 2. **Gathering the Ingredients: Load Data**
We loaded the images and captions like a chef prepping for a feast:
- **Images**: Stored in `/content/flickr8k/Images`.
- **Captions**: A CSV file with image names and their captions (e.g., "A child in a pink dress is climbing up a set of stairs").
- Code snippet:
  ```
  image_path = '/content/flickr8k/Images'
  data = pd.read_csv("/content/flickr8k/captions.txt")
  ```

### 3. **Summoning the Wizards: Import Libraries**
We called upon our trusted allies to help us:
- `TensorFlow` and `Keras` for building the models.
- `DenseNet201` for feature extraction, and `LSTM` for caption generation.
- Code snippet:
  ```
  import tensorflow as tf
  from tensorflow.keras.applications import DenseNet201
  from tensorflow.keras.layers import LSTM, Embedding, Dense
  ```

### 4. **Crafting the Spell: Building the Model**
- **CNN (DenseNet201)**: Extracts features from images, turning pixels into a meaningful “feature vector.”
- **RNN (LSTM)**: Takes the feature vector and generates a caption, word by word, like a poet weaving a story.
- The two work together like a painter and a bard, creating a masterpiece!

### 5. **Casting the Spell: Generating a Caption**
We tested the magic with a sample image:
- Uploaded an image (e.g., `1002674143_1b742ab4b8.jpg`).
- The model worked its magic and generated a caption.
- Code snippet:
  ```
  image_path = "/content/flickr8k/Images/1002674143_1b742ab4b8.jpg"
  generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path)
  ```

## Results: The Magic Unveiled!

Our AI poet worked its charm! Here are some spells it cast:
- **Sample Caption**: "A little girl climbing into a wooden playhouse."
- The model learned to describe scenes with creativity, though it sometimes stumbled (more on that later!).

## Challenges: Dragons We Faced

Wallahi, this journey wasn’t all roses! Some dragons tried to stop us:
- My dataset was tiny ya Gamma, like a small “fool” plate—not enough to fill my model’s tummy! This caused a big “overfitting” mess.
- I only trained for 10 epochs ya3ni, a quick “fawazeer” round, not enough to let my model shine.
- With just 80,000 features, my model was like a young “singer” at “El-Sawy” with only a couple of songs learned!

## Future Magic: Improvements Ahead

Don’t worry ya Gamma, the future is “zay el-foul” (as good as fava beans)! I’ll:
- Hunt for a bigger dataset, like a big “molokhia” pot full of images and captions.
- Train for more epochs—maybe 50 or more—turning it into a “Soad Hosny movie” so my model becomes a “captioning tar” (star).
- Use data augmentation tricks to make my dataset “tasty” without extra “saman” (cost).
- Add regularization “ta3weezat” (charms) like dropout to keep my model grounded—no more “memorizing balaha” (silly date)!

## How to Join the Magic?

Want to try this spell yourself? Here’s how:
1. Clone the repo: `git clone <your-repo-link>`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the Flickr8k dataset using your Kaggle credentials.
4. Run the Jupyter Notebook: `jupyter notebook Image_Caption_Generator.ipynb`.
5. Upload an image and watch the magic unfold!

## Final Words

This project is a love letter to creativity and tech, blending pixels and poetry to make the world a more connected place. Let’s keep weaving stories from images—together! 

*Created with by Mohamed Abolyazeed, May 2025*


# Converting the target environment:
  1. python -m venv venv
  2. source venv/Scripts/activate