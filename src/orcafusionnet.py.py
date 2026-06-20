import torch
import torch.nn as nn
import torchvision.models as models

class OrcaFusionNet25Attention(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # 1. The Visual Brain (ViT)
        self.transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.transformer.encoder.layers[-1].parameters():
            param.requires_grad = True
        self.transformer.heads = nn.Identity()

        # 2. The Math Brain (Processing your 25 features)
        self.meta_fc = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # =======================================================
        # NEW: THE GATED ATTENTION LENS
        # =======================================================
        # We project the 128 metadata features out to exactly 768
        # (which is the exact size of the ViT output).
        # We use Sigmoid to squash every number between 0.0 and 1.0
        self.attention_gate = nn.Sequential(
            nn.Linear(128, 768),
            nn.Sigmoid() 
        )
        # =======================================================

        # 3. The Final Classifier
        # Notice the input is now just 768, not (768 + metadata)
        # because we merged them via multiplication!
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_img, metadata):
        # A. Get the image features: Shape [Batch, 768]
        audio_features = self.transformer(audio_img)
        
        # B. Get the math features: Shape [Batch, 128]
        meta_features = self.meta_fc(metadata)
        
        # C. Generate the Attention Weights (The Volume Knobs): Shape [Batch, 768]
        attention_weights = self.attention_gate(meta_features)
        
        # D. APPLY MULTIMODAL FUSION (Multiply!)
        # The metadata physically alters the visual features here.
        attended_audio = audio_features * attention_weights
        
        # E. Make the final decision based on the focused audio
        return self.classifier(attended_audio)
