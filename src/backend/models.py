import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load a pre-trained ResNet34 model
        base = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        # Add the embedding layer that was in the saved model
        self.embedding = nn.Linear(512, 256)
        # Add the classifier that was in the saved model
        self.classifier = nn.Linear(256, 2)  # Output size is 2
        
    def forward(self, x, return_embedding=False):
        # Extract features
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Apply embedding layer
        embedding = self.embedding(x)
        # Return embedding if requested (for feature extraction)
        if return_embedding:
            return embedding
        # Otherwise, apply classifier and return predictions
        out = self.classifier(embedding)
        return out 