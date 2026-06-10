
import sys
import os
# Adds the 'CausalNet' directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tqdm import tqdm
from data_class_open_ended import get_args
from build_dataset import MSVDDatasetMotion, stgraph_collate
# from models.visualEncoder.new_causal_msvd import STGraphTransformerNet
from models.visualEncoder.msvd_revised_transformer import STGraphTransformerNet
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3

@torch.no_grad()  
def validate_msvd(cfg, model, val_loader, device):
    model.eval()
    print('Validating on MSVD-QA...')
    
    total_correct = 0
    total_count = 0
    
    qtype_names = {0: "what", 1: "who", 2:"how",  3: "when", 4: "where", 5: "other"}
    correct_by_type = {i: 0 for i in qtype_names.keys()}
    count_by_type = {i: 0 for i in qtype_names.keys()}

    for data in tqdm(val_loader):
        
        data = data.to(device)
                
        labels = data["answers"].view(-1)       
        qtype_idx = data["qtype_idx"].view(-1) 

        with torch.autocast(device_type=device.type, enabled=cfg.system.use_amp, dtype=torch.bfloat16):
            out = model(data)
            logits = out["causal_logits"] # Shape: (B, num_answers)
        
        preds = logits.argmax(dim=-1)      
        agreeings = (preds == labels) # Boolean tensor of correct answers
        total_correct += agreeings.sum().item()
        total_count += labels.size(0)

        for q_type in qtype_names.keys():
            type_mask = (qtype_idx == q_type)
            correct_by_type[q_type] += (agreeings & type_mask).sum().item()
            count_by_type[q_type] += type_mask.sum().item()

    overall_acc = (total_correct / total_count) * 100.0 if total_count > 0 else 0.0
    
    print(f"\n--- MSVD-QA Validation Results ---")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_count})")
    
    # Print accuracy for each WH-type
    for q_type, name in qtype_names.items():
        if count_by_type[q_type] > 0:
            type_acc = (correct_by_type[q_type] / count_by_type[q_type]) * 100.0
            print(f" - {name} Acc: {type_acc:.2f}% ({correct_by_type[q_type]}/{count_by_type[q_type]})")
            
    return overall_acc
if __name__ == "__main__":
    import torch

    # 1. Initialize your model architecture
    # (It must be initialized with the exact same config/hyperparameters as when it was trained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_args()
    model = STGraphTransformerNet(cfg).to(device) 
    dataset = MSVDDatasetMotion(cfg, split='test')
    
    ema = ModelEmaV3(
        model,
        decay=0.999
    )
        
    # n_val   = max(1, int(len(dataset) * cfg.data.val_split))
    # n_train = len(dataset) - n_val
    # train_ds, val_ds = random_split(
    #     dataset,
    #     [n_train, n_val],
    #     generator=torch.Generator().manual_seed(cfg.system.seed),
    # )
    print(f"Test videos : {len(dataset)}")
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
    best_pt = '/root/autodl-tmp/CausalSTGNet/train/msvd_motion_configs/msvd_motion_layer3_configs/casual_1/layer2_feature/no_hstr/best_acc.pt'
    checkpoint = torch.load(
        best_pt,
        map_location=device,
        weights_only=False
    )

    # =====================================================
    # Load normal model
    # =====================================================

    model.load_state_dict(
        checkpoint["model"]
    )

    print(
        f"✅ Loaded model from epoch "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    # =====================================================
    # Load EMA weights
    # =====================================================

    if "ema" in checkpoint and checkpoint["ema"] is not None:

        ema.load_state_dict(
            checkpoint["ema"]
        )

        eval_model = ema.module

        print("✅ EMA weights loaded")

    else:

        eval_model = model

        print("⚠️ EMA weights not found, using raw model")

    # =====================================================
    # Evaluation mode
    # =====================================================

    eval_model.eval()

    # =====================================================
    # Validation
    # =====================================================

    validate_msvd(
        cfg,
        eval_model,
        val_loader,
        device
    )
    # checkpoint = torch.load(best_pt, map_location=device, weights_only=False)

    # # 3. Load the weights into the model
    #     # 3. Load the weights into the model
    # if 'model' in checkpoint:
    #     # Your training script saved the weights under the key 'model'
    #     model.load_state_dict(checkpoint['model'])
    #     saved_epoch = checkpoint.get('epoch', 'Unknown')
    #     print(f"✅ Successfully loaded weights from Epoch {saved_epoch}-accuracy {checkpoint.get('best_acc', 'Unknown')}")
    # elif 'model_state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     saved_epoch = checkpoint.get('epoch', 'Unknown')
    #     print(f"✅ Successfully loaded weights from Epoch {saved_epoch}")
    # else:
    #     model.load_state_dict(checkpoint)
    #     print("✅ Successfully loaded raw model weights")

    # # 4. Set the model to Evaluation Mode (CRITICAL: disables Dropout and sets BatchNorm correctly)
    # model.eval()

    # # 5. Run the validation function!
    # validate_msvd(cfg, model, val_loader, device)