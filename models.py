"""
This script defines the neural network architectures.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use a pre-trained ResNet as the encoder
        resnet = models.resnet50(pretrained=True)
        # Remove the last fully connected layer (classification layer)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a fully connected layer to map ResNet's output to the desired embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Ensure that pretrained weights are frozen initially for faster training and stability
        # We can unfreeze them later or for fine-tuning if needed.
        # If we want to fine-tune the entire ResNet model, either remove this block or conditionally enable/disable it
        with torch.no_grad():  # Freeze CNN weights during training initially
            features = self.resnet(images)
        # end of this block
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Linear layer to transform image features into initial hidden and cell states for LSTM
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions, lengths):
        # Embed captions
        embeddings = self.embed(captions)

        # Initialize LSTM's hidden and cell states from image features
        # Unsqueeze(0) to add the num_layers dimension
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, hidden_size)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, hidden_size)

        # Pack padded sequence for efficient RNN training
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        hiddens, _ = self.lstm(packed, (h0, c0))
        outputs = self.linear(hiddens[0])  # hiddens[0] contains the unpacked output
        return outputs

    # Simple greedy sample method for inference
    # Try Beam search optimization in future
    def sample(self, features, max_len=50):
        """Generates captions for given image features using greedy search."""
        sampled_ids = []

        # Initialize LSTM's hidden and cell states from image features
        h = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Start with <start> token embedding as the first input to the LSTM
        # Input shape for LSTM: (batch_size, sequence_length, embed_size)
        # Here, batch_size is 1 for a single image, sequence_length is 1 for one word at a time.
        inputs = self.embed(torch.tensor([[1]], dtype=torch.long).to(features.device))  # 1 is <start> token ID

        for i in range(max_len):
            # Pass input and states to LSTM
            hiddens, (h, c) = self.lstm(inputs, (h, c))  # hiddens: (1, 1, hidden_size)

            # Predict the next word
            outputs = self.linear(hiddens.squeeze(1))  # Remove sequence length dim: (1, vocab_size)

            # Get the word with the highest probability
            _, predicted = outputs.max(1)  # predicted: (1)

            sampled_ids.append(predicted.item())

            # If <end> token is predicted, stop generating
            if predicted.item() == 2:  # vocab["<end>"] is 2
                break

            # Use the predicted word's embedding as the input for the next step
            inputs = self.embed(predicted.unsqueeze(0))  # Add sequence dim: (1, 1, embed_size)

        return sampled_ids