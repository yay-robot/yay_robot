import torch
import torch.nn as nn
import numpy as np
from clip import load
import torchvision.transforms as transforms
import random

clip_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ONE_HOT = False  # ablation TODO: add argument


class Instructor(nn.Module):
    def __init__(
        self,
        device,
        history_len,
        output_size=768,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        candidate_embeddings=None,
        candidate_texts=None,
        command_to_index=None,
        num_cameras=4,
    ):
        super().__init__()

        # Load the pretrained CLIP model
        self.clip_model, self.clip_text_model = load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False  # Freeze the CLIP model parameters

        # Transformer for processing sequences of image embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.clip_model.visual.output_dim,
                nhead=num_heads,
                dim_feedforward=hidden_size,
            ),
            num_layers=num_layers,
        )

        if ONE_HOT:
            output_size = len(candidate_texts)

        self.mlp = nn.Sequential(
            nn.Linear(self.clip_model.visual.output_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))

        # Positional Encoding
        self.positional_encoding = self.create_sinusoidal_embeddings(
            self.clip_model.visual.output_dim, (history_len + 1) * num_cameras
        )

        self.history_len = history_len
        self.candidate_embeddings = candidate_embeddings
        self.candidate_texts = candidate_texts
        self.command_to_index = command_to_index

        total, trainable = count_parameters(self)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

    def forward(self, images):
        # Given images of shape (b, t, k, c, h, w)
        batch_size, timesteps, num_cameras, c, h, w = images.shape

        # Check if padding is required
        if timesteps < self.history_len + 1:
            padding_needed = self.history_len + 1 - timesteps
            padding = torch.zeros(
                (batch_size, padding_needed, num_cameras, c, h, w), device=images.device
            )
            images = torch.cat([padding, images], dim=1)
            timesteps = (
                self.history_len + 1
            )  # Update timesteps to reflect the new length

        # Reshape images to (b*t*k, c, h, w) for processing through CLIP
        images_reshaped = images.reshape(batch_size * timesteps * num_cameras, c, h, w)

        # Apply transformations for CLIP
        images_transformed = clip_transform(
            images_reshaped
        )  # CLIP model expects images to be normalized and resized to 224*224

        # Get image features from CLIP
        image_features = self.clip_model.encode_image(images_transformed)

        # Reshape the image features to [batch_size, timesteps*cameras, feature_dim]
        image_features_reshaped = image_features.reshape(
            batch_size, timesteps * num_cameras, -1
        ).to(torch.float32)

        # Add positional encoding
        image_features_reshaped += self.positional_encoding[
            : timesteps * num_cameras, :
        ].to(image_features_reshaped.device)

        # Pass the concatenated features through the Transformer
        transformer_out = self.transformer(
            image_features_reshaped.transpose(0, 1)
        ).transpose(0, 1)

        # Extract the final output of the Transformer for each sequence in the batch
        final_output = transformer_out[:, -1, :]

        if ONE_HOT:
            # Directly predict the logits for each command
            logits = self.mlp(final_output)
        else:
            # Predict the command embedding
            command_pred = self.mlp(final_output)
            # Compute the similarity scores as logits
            logits = self.compute_similarities(command_pred) / self.temperature.clamp(
                min=1e-8
            )

        return logits, self.temperature

    def compute_similarities(self, embeddings):
        # Compute the cosine similarities
        cosine_similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )

        return cosine_similarities

    @staticmethod
    def create_sinusoidal_embeddings(d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def decode_logits(self, logits, temperature):
        # Compute the probabilities
        probs = (
            logits
            if ONE_HOT
            else torch.nn.functional.softmax(logits / temperature, dim=-1)
        )

        # Find the indices of the max logit for each example in the batch
        _, max_indices = torch.max(probs, dim=-1)

        return [self.candidate_texts[index] for index in max_indices.cpu().numpy()]

    def get_nearest_text(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Map the indices back to the actual texts
        return [self.candidate_texts[i] for i in indices.cpu().numpy()]

    def get_nearest_embedding(self, embeddings):
        # Compute cosine similarities
        similarities = self.compute_similarities(embeddings)

        # Get the index of the maximum similarity for each prediction
        indices = similarities.argmax(dim=-1)

        # Print the top 5 candidates
        probs = torch.nn.functional.softmax(similarities, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], 5)
        normalized_top_probs = top_probs / top_probs.sum()
        for i, (index, prob) in enumerate(zip(top_indices, normalized_top_probs)):
            print(
                f"Candidate {i}: {self.candidate_texts[index]}, Normalized Prob: {prob:.4f}"
            )

        # Map the indices back to the embeddings
        return [self.candidate_embeddings[i] for i in indices.cpu().numpy()]

    def get_random_from_top_k(self, embeddings, k=3):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        top_k_indices = similarities.topk(k, dim=-1)[1]

        # Randomly select one from the top-k for each row
        selected_indices = [
            random.choice(indices_row) for indices_row in top_k_indices.cpu().numpy()
        ]

        return [self.candidate_texts[i] for i in selected_indices]

    def sample_with_temperature(self, embeddings, temperature=1.0):
        similarities = (embeddings @ self.candidate_embeddings.T) / (
            embeddings.norm(dim=-1, keepdim=True)
            * self.candidate_embeddings.norm(dim=-1, keepdim=True).T
        )
        probs = torch.nn.functional.softmax(similarities / temperature, dim=-1)
        sampled_indices = torch.multinomial(
            probs, 1
        ).squeeze()  # Squeezing to potentially remove singleton dimensions
        # Check if sampled_indices is a scalar (0-dim) or an array
        if sampled_indices.ndim == 0:
            # If it's a scalar, we make it a one-element array
            sampled_indices = [sampled_indices.item()]
        else:
            # Otherwise, we convert it to a list
            sampled_indices = sampled_indices.tolist()

        return [self.candidate_texts[i] for i in sampled_indices]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# Example usage:
if __name__ == "__main__":
    from dataset import load_data

    # Dataset and Dataloader parameters
    dataset_dir = "/scr/lucyshi/dataset/aloha_bag_3_objects"
    num_episodes = 10
    camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    batch_size_train = 8
    batch_size_val = 8

    # Load the dataloader
    train_dataloader, val_dataloader, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
    )

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Instructor(device=device, history_len=5)
    model.to(device)

    # Fetch a batch of data and pass it through the model
    for image_data, language_data, _ in train_dataloader:
        image_data = image_data.to(device)
        predictions = model(image_data)
        print(f"Image data shape: {image_data.shape}")
        print(f"Language data shape: {language_data.shape}")
        print(f"Predictions shape: {predictions.shape}")
        break
