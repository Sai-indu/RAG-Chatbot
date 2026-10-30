from sentence_transformers import SentenceTransformer
import os

# Download model
print("Downloading embeddings model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Save to local directory
os.makedirs('./models', exist_ok=True)
model.save('./models/all-MiniLM-L6-v2')
print("✓ Model saved to ./models/all-MiniLM-L6-v2")
print("✓ You can now run the chatbot offline!")