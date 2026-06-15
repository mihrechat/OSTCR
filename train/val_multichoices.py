
import sys
import os
# Adds the 'CausalNet' directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tqdm import tqdm
from models.VisualEncoder.tgif_trans import *
# from models.visualEncoder.no_causal_stud_motion import STGraphTransformerNet
from data_class_multi_choice import get_args
from build_dataset import STUDTrafficDatasetMotion, stgraph_collate
from torch.utils.data import DataLoader, random_split


@torch.no_grad()
def validate(cfg, model, val_loader, device):
    model.eval()
    
    # 1. Track Overall Accuracy
    total_correct = 0
    total_count = 0
    
    # 2. Track Accuracy by Question Type
    num_qtypes = 6  # Update this to match your dataset's number of question types
    correct_by_type = {i: 0 for i in range(num_qtypes)}
    count_by_type = {i: 0 for i in range(num_qtypes)}
    # q_types = {0: "u", 1: "a", 2:"c", 3: "r", 4: "i", 5: "f"}
    q_types = {0: "u" , 1: "a", 2: "f",3: "i", 4:"r",5: "c"}

    print('Validating...')
    for data in tqdm(val_loader):
        
        data = data.to(device)
        labels = data["answers"]       # The correct option index
        qtype_idx = data["qtype_idx"].squeeze(-1) # The integer representing question type

        # ── Forward Pass ──
        out = model(data)
        causal_logits = out["causal_logits"] # (B, 4)
        
        # ── Calculate Predictions ──
        preds = causal_logits.argmax(dim=-1)      
        agreeings = (preds == labels)             

        # ── 1. Update Overall Accuracy ──
        total_correct += agreeings.sum().item()
        total_count += labels.size(0)

        # ── 2. Update Per-Type Accuracy (Vectorized, NO FOR-LOOPS!) ──
        for q_type in range(num_qtypes):
            # Create a mask for just this question type
            type_mask = (qtype_idx == q_type)
            
            # Count how many of THIS type were correct, and how many existed in the batch
            correct_by_type[q_type] += (agreeings & type_mask).sum().item()
            count_by_type[q_type] += type_mask.sum().item()

    # ── Final Calculations ──
    overall_acc = (total_correct / total_count) * 100.0 if total_count > 0 else 0.0
    
    print(f"\n--- Validation Results ---")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    
    # Print accuracy for each type
    # type_names = ["Type 0", "Type 1", "Type 2", "Type 3", "Type 4", "Type 5"] # Replace with real names!
    for q_type in range(num_qtypes):
        if count_by_type[q_type] > 0:
            type_acc = (correct_by_type[q_type] / count_by_type[q_type]) * 100.0
            print(f" - {q_types[q_type]} Acc: {type_acc:.2f}% ({correct_by_type[q_type]}/{count_by_type[q_type]})")
            
    return overall_acc
   
# import torch
# from tqdm import tqdm

# @torch.no_grad()
# def validate(cfg, model, val_loader, device):
#     model.eval()

#     total_correct = 0
#     total_count = 0

#     # dynamic containers (no assumption on number of classes)
#     correct_by_type = {}
#     count_by_type = {}

#     print("Validating...")

#     for data in tqdm(val_loader):

#         # move tensors safely
#         data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

#         labels = data["answers"].long()
#         qtype_idx = data["qtype_idx"].view(-1)

#         # optional but useful if you want human-readable names
#         qtype_name = data.get("qtype_name", None)

#         out = model(data)
#         causal_logits = out["causal_logits"]

#         preds = causal_logits.argmax(dim=-1)
#         agreeings = (preds == labels)

#         # overall stats
#         total_correct += agreeings.sum().item()
#         total_count += labels.size(0)

#         # per-type stats (fully dynamic)
#         for i in range(labels.size(0)):
#             qt = int(qtype_idx[i].item())

#             if qt not in correct_by_type:
#                 correct_by_type[qt] = 0
#                 count_by_type[qt] = 0

#             count_by_type[qt] += 1
#             correct_by_type[qt] += int(agreeings[i].item())

#     # overall accuracy
#     overall_acc = 100.0 * total_correct / total_count if total_count > 0 else 0.0

#     print("\n--- Validation Results ---")
#     print(f"Overall Accuracy: {overall_acc:.2f}%")

#     # print per-type results (sorted for readability)
#     for qt in sorted(count_by_type.keys()):
#         acc = 100.0 * correct_by_type[qt] / count_by_type[qt]
#         print(f" - Type {qt}: {acc:.2f}% ({correct_by_type[qt]}/{count_by_type[qt]})")

#     return overall_acc

if __name__ == "__main__":
    import torch

    # 1. Initialize your model architecture
    # (It must be initialized with the exact same config/hyperparameters as when it was trained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_args()
    cfg.model.test = True
    print(f'--------Testing model----------------: {cfg.model.test}')
    model = STGraphTransformerNet(cfg).to(device) # import the corresponding models for validation
    dataset = STUDTrafficDatasetMotion(cfg, "test") #
        
    # n_val   = max(1, int(len(dataset) * cfg.data.val_split))
    # n_train = len(dataset) - n_val
    # train_ds, val_ds = random_split(
    #     dataset,
    #     [n_train, n_val],
    #     generator=torch.Generator().manual_seed(cfg.system.seed),
    # )
    print(f"Train videos : {len(dataset)}")
    val_loader = DataLoader(
        dataset,
        batch_size  = cfg.train.batch_size,
        shuffle     = False,
        num_workers = cfg.data.num_workers,
        collate_fn  = stgraph_collate,
        pin_memory  = device.type == "cuda",
    )

    # 2. Load the saved checkpoint file
    print("Loading best.pt checkpoint...")
    best_pt = '/root/autodl-tmp/CausalSTGNet/train/stud_logs/stud_causal/stud_causal2/best_acc.pt'
    checkpoint = torch.load(best_pt, map_location=device, weights_only=False)

    # 3. Load the weights into the model
        # 3. Load the weights into the model
    if 'model' in checkpoint:
        # Your training script saved the weights under the key 'model'
        model.load_state_dict(checkpoint['model'])
        saved_epoch = checkpoint.get('epoch', 'Unknown')
        print(f"Successfully loaded weights from Epoch {saved_epoch}")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint.get('epoch', 'Unknown')
        print(f"Successfully loaded weights from Epoch {saved_epoch}")
    else:
        model.load_state_dict(checkpoint)
        print(" Successfully loaded raw model weights")

    # 4. Set the model to Evaluation Mode (CRITICAL: disables Dropout and sets BatchNorm correctly)
    model.eval()

    # 5. Run the validation function!
    # (Assuming you have your validation dataloader ready as `val_loader`)
    validate(cfg, model, val_loader, device)
