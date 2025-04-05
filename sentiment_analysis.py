from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F

# Load the FinBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Example financial text to analyze
text = (
    "Given the recent downturn in stocks especially in tech which is likely to persist as yields keep going up, "
    "I thought it would be prudent to share the risks of investing in ARK ETFs, written up very nicely by "
    "[The Bear Cave](https://thebearcave.substack.com/p/special-edition-will-ark-invest-blow). The risks come "
    "primarily from ARK's illiquid and very large holdings in small cap companies. ARK is forced to sell its "
    "holdings whenever its liquid ETF gets hit with outflows as is especially the case in market downturns. "
    "This could force very painful liquidations at unfavorable prices and the ensuing crash goes into a "
    "positive feedback loop leading into a death spiral enticing even more outflows and predatory shorts."
)

# Tokenize the input text
tokens = tokenizer.encode_plus(
    text,
    max_length=512,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='pt'  # Return PyTorch tensors
)

# Run the model to get raw predictions (logits)
output = model(**tokens)

# Convert logits to probabilities using softmax
probs = F.softmax(output.logits, dim=-1)

# Get the predicted class (0 = negative, 1 = neutral, 2 = positive)
predicted_class = torch.argmax(probs, dim=1).item()

# Print the result
print("Sentiment Probabilities:", probs)
print("Predicted Sentiment Class:", predicted_class)
