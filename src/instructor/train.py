"""
Example usage:

python instructor/train.py \
    --task_name aloha_bag_3_objects aloha_bag_3_objects_d1_v0 aloha_bag_3_objects_d1_v1 aloha_bag_3_objects_d1_v2 \
    --ckpt_dir /scr/lucyshi/hl_ckpt/aloha_bag_v0_v1_v2_his_2_sf_50_offset_10_lr \
    --batch_size 16 --num_epochs 1000  --lr 1e-4 \
    --seed 0 --gpu 1 --test_only 

"""
import torch
import torch.optim as optim
import argparse
import os
import numpy as np
import wandb
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import threading
import sys
sys.path.append("$PATH_TO_YAY_ROBOT/src")  # to import aloha
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from collections import OrderedDict

from instructor.dataset import load_merged_data
from instructor.model import Instructor
from aloha_pro.aloha_scripts.utils import memory_monitor


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images, _, commands = batch
        images = images.to(device)

        optimizer.zero_grad()
        logits, temperature = model(images)

        # Convert ground truth command strings to indices using the pre-computed dictionary
        commands_idx = [
            model.command_to_index[
                cmd.replace("the back", "the bag").replace("mmove", "move")
            ]
            for cmd in commands
        ]
        commands_idx = torch.tensor(commands_idx, device=device)

        loss = criterion(logits, commands_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if args.log_wandb:
            wandb.log({"Train Loss": loss.item(), "Temperature": temperature.item()})
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images, _, commands = batch
            images = images.to(device)

            logits, temperature = model(images)

            # Convert ground truth command strings to indices using the pre-computed dictionary
            commands_idx = [
                model.command_to_index[cmd.replace("the back", "the bag")]
                for cmd in commands
            ]
            commands_idx = torch.tensor(commands_idx, device=device)

            loss = criterion(logits, commands_idx)
            total_loss += loss.item()

            if args.log_wandb:
                wandb.log({"Eval Loss": loss.item(), "Temperature": temperature.item()})
                # wandb.log({"Eval Loss": loss.item()})
    return total_loss / len(dataloader)


def test(model, dataloader, device, current_epoch):
    model.eval()

    total_correct = 0
    total_predictions = 0

    # predicted_embeddings = []
    # gt_embeddings = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images, command_embedding_gt, command_gt = batch
            images = images.to(device)

            logits, temperature = model(images)
            # Get nearest text for each prediction in the batch
            decoded_texts = model.decode_logits(logits, temperature)

            # predicted_embeddings.extend(predictions.cpu().numpy())
            # gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            for i, (gt, pred) in enumerate(zip(command_gt, decoded_texts)):
                # Save incorrect prediction
                # if pred != gt:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_incorrect_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)
                #     if args.log_wandb:
                #         wandb.log({f"Incorrect Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {idx}, Image {i}")})
                # elif i < 5:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_correct_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)

                total_correct += int(pred == gt)
                total_predictions += 1
                print(f"Ground truth: {gt} \t Predicted text: {pred}")

    # Visualize embeddings
    # tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, current_epoch)

    success_rate = total_correct / total_predictions
    print(f"Epoch {current_epoch}: Success Rate = {success_rate * 100:.2f}%")

    if args.log_wandb:
        wandb.log({"Success Rate": success_rate})

    return success_rate


def latest_checkpoint(ckpt_dir):
    """
    Returns the latest checkpoint file from the given directory.
    """
    all_ckpts = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".ckpt")
    ]
    epoch_numbers = [int(f.split("_")[1].split(".")[0]) for f in all_ckpts]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_idx = max(epoch_numbers)
    return os.path.join(ckpt_dir, f"epoch_{latest_idx}.ckpt"), latest_idx


def load_candidate_texts(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Extract the instruction (text before the colon), strip whitespace, and then strip quotation marks
        candidate_texts = [line.split(":")[0].strip().strip("'\"") for line in lines]
    return candidate_texts


def save_combined_image(image, gt_text, pred_text, save_path=None):
    # image = image[:, :, [2, 1, 0]]

    # Extract first frame t=0 and concatenate across width
    combined_image = torch.cat([image[0, i] for i in range(image.shape[1])], dim=-1)

    # Convert to PIL image
    combined_image_pil = transforms.ToPILImage()(combined_image)

    # Create a blank canvas to add text
    canvas = Image.new(
        "RGB", (combined_image_pil.width, combined_image_pil.height + 100), "black"
    )
    canvas.paste(combined_image_pil, (0, 100))

    # Add GT and predicted text
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30
    )
    draw.text((10, 10), "GT: " + gt_text, font=font, fill="white")
    draw.text((10, 50), "Pred: " + pred_text, font=font, fill="red")

    if save_path is not None:
        canvas.save(save_path)
    else:
        return canvas


def tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, epoch):
    # Convert lists to numpy arrays
    predicted_embeddings = np.array(predicted_embeddings)
    gt_embeddings = np.array(gt_embeddings)

    assert (
        predicted_embeddings.shape == gt_embeddings.shape
    ), "The number of predicted and ground truth embeddings do not match."

    # Stack embeddings and apply t-SNE
    all_embeddings = np.vstack(
        [predicted_embeddings, gt_embeddings, candidate_embeddings.cpu().numpy()]
    )
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split the 2D embeddings back
    predicted_2d = embeddings_2d[: len(predicted_embeddings)]
    gt_2d = embeddings_2d[
        len(predicted_embeddings) : len(predicted_embeddings) + len(gt_embeddings)
    ]
    candidate_2d = embeddings_2d[len(predicted_embeddings) + len(gt_embeddings) :]

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.scatter(
        candidate_2d[:, 0], candidate_2d[:, 1], marker="o", color="g", label="Dataset"
    )
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], marker="o", color="b", label="Ground Truth")
    plt.scatter(
        predicted_2d[:, 0], predicted_2d[:, 1], marker="o", color="r", label="Predicted"
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    plt.legend()

    # Save with the epoch in the filename
    image_save_path = os.path.join(ckpt_dir, f"embeddings_tsne_epoch_{epoch}.png")
    plt.savefig(image_save_path)

    # Log the image to wandb if logging is enabled
    if args.log_wandb:
        wandb.log(
            {
                "t-SNE Visualization": [
                    wandb.Image(image_save_path, caption=f"Epoch {epoch}")
                ]
            }
        )


def load_candidate_texts_and_embeddings(dataset_dirs, device=torch.device("cuda")):
    candidate_texts = []
    candidate_embeddings = []

    for dataset_dir in dataset_dirs:
        embeddings_path = os.path.join(
            dataset_dir, "candidate_embeddings_distilbert.npy"
        )
        # Load pre-computed embeddings
        candidate_embedding = (
            torch.tensor(np.load(embeddings_path).astype(np.float32))
            .to(device)
            .squeeze()
        )
        candidate_embeddings.append(candidate_embedding)
        candidate_texts_path = os.path.join(dataset_dir, "count.txt")
        current_candidate_texts = load_candidate_texts(candidate_texts_path)
        candidate_texts.extend(current_candidate_texts)
    candidate_embeddings = torch.cat(candidate_embeddings, dim=0).to(device)

    def remove_duplicates(candidate_texts, candidate_embeddings):
        unique_entries = OrderedDict()

        for text, embedding in zip(candidate_texts, candidate_embeddings):
            if text not in unique_entries:
                unique_entries[text] = embedding

        # Rebuild the lists without duplicates
        filtered_texts = list(unique_entries.keys())
        filtered_embeddings = torch.stack(list(unique_entries.values()))

        return filtered_texts, filtered_embeddings

    candidate_texts, candidate_embeddings = remove_duplicates(
        candidate_texts, candidate_embeddings
    )
    return candidate_texts, candidate_embeddings


def build_instructor(dataset_dirs, history_len, device):
    # Load candidate texts and embeddings
    candidate_texts, candidate_embeddings = load_candidate_texts_and_embeddings(
        dataset_dirs, device=device
    )
    command_to_index = {command: index for index, command in enumerate(candidate_texts)}

    # Build model
    model = Instructor(
        device=device,
        history_len=history_len,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
    ).to(device)
    return model


if __name__ == "__main__":
    threading.Thread(target=memory_monitor, daemon=True).start()

    parser = argparse.ArgumentParser(description="Train and evaluate command prediction model using CLIP.")
    parser.add_argument('--task_name', nargs='+', type=str, help='List of task names', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=1)
    parser.add_argument('--prediction_offset', action='store', type=int, help='prediction_offset', default=20)
    parser.add_argument('--history_skip_frame', action='store', type=int, help='history_skip_frame', default=50)
    parser.add_argument('--test_only', action='store_true', help='Test the model using the latest checkpoint and exit')
    parser.add_argument('--random_crop', action='store_true')
    parser.add_argument('--dagger_ratio', action='store', type=float, help='dagger_ratio', default=None)

    args = parser.parse_args()

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # get task parameters
    from aloha_pro.aloha_scripts.constants import TASK_CONFIGS

    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in args.task_name:
        task_config = TASK_CONFIGS[task]
        dataset_dirs.append(task_config["dataset_dir"])
        num_episodes_list.append(task_config["num_episodes"])
        max_episode_len = max(max_episode_len, task_config["episode_len"])
        camera_names = task_config["camera_names"]
    ckpt_dir = args.ckpt_dir
    dagger_ratio = args.dagger_ratio

    # Data loading
    train_dataloader, val_dataloader, test_dataloader = load_merged_data(
        dataset_dirs=dataset_dirs,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        history_len=args.history_len,
        prediction_offset=args.prediction_offset,
        history_skip_frame=args.history_skip_frame,
        random_crop=args.random_crop,
        dagger_ratio=dagger_ratio,
    )

    model = build_instructor(dataset_dirs, args.history_len, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # WandB initialization
    if args.log_wandb:
        run_name = "instructor." + ckpt_dir.split("/")[-1] + f".{args.seed}"
        wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
        # check if it exists
        if os.path.exists(wandb_run_id_path):
            with open(wandb_run_id_path, "r") as f:
                saved_run_id = f.read().strip()
            wandb.init(
                project="yay-robot", entity="$WANDB_ENTITY", name=run_name, resume=saved_run_id
            )
        else:
            wandb.init(
                project="yay-robot",
                entity="$WANDB_ENTITY",
                name=run_name,
                config=args,
                resume="allow",
            )
            # Ensure the directory exists before trying to open the file
            os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
            with open(wandb_run_id_path, "w") as f:
                f.write(wandb.run.id)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        latest_idx = 0
    else:
        # Load the most recent checkpoint if available
        latest_ckpt, latest_idx = latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            print(f"Loading checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        else:
            print("No checkpoint found.")
            latest_idx = 0

    predictions_dir = os.path.join(ckpt_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if args.test_only:
        test(model, test_dataloader, device, latest_idx)
        exit()

    # Training loop
    pbar_epochs = tqdm(range(latest_idx, args.num_epochs), desc="Epochs")
    for epoch in pbar_epochs:
        # Test the model and log success rate every 200 epochs
        if epoch % 200 == 0 and (epoch > 0 or dagger_ratio is not None):
            test(model, test_dataloader, device, epoch)

        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        if dagger_ratio is None:
            eval_loss = evaluate(model, val_dataloader, criterion, device)

        pbar_epochs.set_postfix({"Train Loss": train_loss})

        if args.log_wandb:
            wandb.log({"Epoch Train Loss": train_loss}, step=epoch)
            if dagger_ratio is None:
                wandb.log({"Epoch Eval Loss": eval_loss}, step=epoch)

        # Save a checkpoint every 100 epochs
        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt")
            torch.save(model.state_dict(), ckpt_path)

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of prune_freq epochs
            prune_freq = 300
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % prune_freq != 0:
                prune_path = os.path.join(ckpt_dir, f"epoch_{prune_epoch}.ckpt")
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    if args.log_wandb:
        wandb.finish()
