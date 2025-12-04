mart Wardrobe System 👔👗
Ever stood in front of your closet wondering what goes well together? This project solves that problem using AI! It's a fashion recommendation system that learns what clothing items look good together and helps you build better outfits.
What does it do?
Give the system any piece of clothing (say, a shirt), and it'll search through your wardrobe to find the best matching pants, shoes, or accessories. It's trained on thousands of real fashion outfits, so it actually understands style compatibility.
Cool Features

Finds matching items from different clothing categories
Shows you the top 5 best matches with confidence scores
Can build complete outfits by matching across multiple categories
Works with 18+ clothing types - from basics like shirts and pants to accessories like watches and bags
Handles large wardrobes (tested with 10,000+ items per category)

How It Works
The system uses something called a Siamese Neural Network trained with triplet loss. Basically, it learns by looking at:

An anchor item (like a pair of pants)
A positive match (a shirt that goes well with those pants)
A negative example (a shirt that doesn't match)

Over time, it figures out patterns in what makes outfits work together. The model creates a 256-dimensional "style fingerprint" for each clothing item, and items that go well together have similar fingerprints.
The Dataset
I used the Re-PolyVore dataset, which has real fashion items organized into complete outfits. It includes:

Bags, bracelets, dresses, pants, shoes, skirts, tops, watches
Plus accessories like hats, eyewear, earrings, necklaces
Thousands of images per category

The dataset structure looks like this:
Re-PolyVore/
├── top/
├── pants/
├── shoes/
├── bag/
└── ... (14 more categories)
Each category folder is filled with images of items from that category.
What you'll need:

Python 3.7 or newer
TensorFlow 2.x
About 4GB of RAM (more if you're loading huge wardrobes)
A GPU helps but isn't required for inference

Installation:
bash# Clone this repo
git clone https://github.com/yourusername/smart-wardrobe-system.git
cd smart-wardrobe-system

# Install what you need
pip install tensorflow numpy matplotlib pandas scikit-learn pillow
Get the dataset: The link of Re-Polyvore Dataset is https://drive.google.com/file/d/1nKmWmQOidUsZ8zVq09vt1_DzUdY1cEjH/view?usp=sharing
Download the Re-PolyVore dataset and put it in the project folder. Your structure should look like:
smart-wardrobe-system/
├── Re-PolyVore/
│   ├── top/
│   ├── pants/
│   └── ...
├── train_model.py
├── inference.py
└── README.md
Training Your Own Model
If you want to train from scratch (takes a few hours on a decent GPU):
bashpython train_model.py
This will:

Load the dataset
Create training pairs of matching/non-matching items
Train the neural network for 10 epochs
Save the model as siameese_embedding.h5

The training creates positive pairs from items in the same outfit and negative pairs from items in different outfit styles. After 10 epochs, the model usually converges to around 90%+ accuracy in distinguishing compatible vs incompatible items.
Using the System
Once you have a trained model, run:
bashpython inference.py
The script will walk you through:

Which clothing categories you want to load (you can pick specific ones or load everything)
Whether to load all items or limit to a smaller set (useful for testing)
Your input image path
Whether you want matches for one category or a complete outfit

Example session:
Enter path to your clothing item: my_blue_shirt.jpg

Matching Options:
1. Match with specific category (see top 5 matches)
2. Get complete outfit (best match from each category)

Enter your choice: 1
Enter category to match with: pants

Searching through 10000 pants items...
Found 5 matches

TOP MATCHES:
1. dark_jeans_001.jpg - Similarity: 0.8234
2. black_trousers_045.jpg - Similarity: 0.8156
3. navy_chinos_012.jpg - Similarity: 0.8098
...
About the Models
Why aren't the model files included?
The trained models are around 90MB each, which is too big for GitHub. You can either:

Train your own (takes a few hours). Use GPU always.
Download pre-trained models from [https://drive.google.com/file/d/1OTZ5urUXXkL8jtTf5ItQe2-C5tQWyR7G/view?usp=drive_link,https://drive.google.com/file/d/17XKR6ySrwHDy32JYT4o2LyBlSgmDTlp8/view?usp=sharing]

Model files you'll need:

siameese_embedding.h5 - This is the main one (about 90MB)
siamese_model.h5 - Full model with training components (optional, about 180MB)

Project Structure
smart-wardrobe-system/
├── train_model.py              # Training script
├── inference.py                # Run outfit matching
├── Re-PolyVore/               # Dataset folder
├── siameese_embedding.h5      # Trained model (you'll create this)
├── re-poly-clustered.csv      # Dataset metadata
└── README.md                  # You're reading it!
Model Architecture Details
For the curious, here's what's under the hood:
Base: ResNet50 pretrained on ImageNet (frozen until conv5_block1_out)
Embedding layers:

Flatten
Dense 512 → ReLU → BatchNorm
Dense 256 → ReLU → BatchNorm
Dense 256 (final embedding)

Training setup:

Adam optimizer, learning rate 0.0001
Batch size 32
Triplet margin 0.5
Input images 200x200 pixels

The embedding layer outputs a 256-dimensional vector for each image. During training, the triplet loss pulls compatible items closer and pushes incompatible ones apart in this 256D space.
Results
After 10 epochs of training, the model achieves:

Triplet loss converges to around 0.1-0.2
Good visual matching results - items that "look good together" get high similarity scores
Top-5 accuracy: the correct matching item is usually in the top 5 recommendations

The cosine similarity scores typically range from:

0.7-0.9 for highly compatible items
0.4-0.6 for neutral matches
Below 0.4 for clashing items

For better matching results:

Use clear, well-lit photos of individual items
Remove backgrounds if possible
Keep the same image quality as your training data

If inference is slow:

Load fewer items per category (500-1000 is usually enough)
Use a GPU if you have one
Pre-compute embeddings for your wardrobe and save them

If you're getting weird matches:

Your model might need more training epochs
Try adjusting the triplet margin
Make sure your input images are preprocessed the same way as training data
