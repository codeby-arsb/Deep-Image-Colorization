import os
import sys
import time
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

import random
import numpy as np

from model import ColorizationUNet
from dataset import create_dataloader
from device import get_device, get_device_name, get_amp_device_type, get_memory_stats
from evaluate import evaluate_fixed_test_images
from losses import get_loss_function, get_loss_metadata, DEFAULT_SMOOTH_L1_BETA

def get_args():
    parser = argparse.ArgumentParser(description="Train Colorization U-Net")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smooth_l1"],
                        help="Loss function: 'mse' (Baseline) or 'smooth_l1' (Huber)")
    parser.add_argument("--loss-beta", type=float, default=DEFAULT_SMOOTH_L1_BETA,
                        help="Beta threshold parameter for Smooth L1 loss (default: 1.0)")
    parser.add_argument("--exp-dir", type=str, default=None,
                        help="Custom experiment output directory for checkpoints and metrics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke test instead of full training")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    
    device = get_device()
    amp_type = get_amp_device_type(device)
    use_amp = args.amp and (amp_type is not None)
    
    # Loss, Optimizer, Scheduler
    criterion = get_loss_function(args.loss, beta=args.loss_beta)
    loss_meta = get_loss_metadata(args.loss, beta=args.loss_beta)
    
    # Output paths with strict baseline protection
    if args.exp_dir:
        exp_dir = os.path.abspath(args.exp_dir)
        checkpoints_dir = os.path.join(exp_dir, "checkpoints")
        plots_dir = os.path.join(exp_dir, "plots")
        history_file = os.path.join(exp_dir, "training_history.csv")
        eval_dir = os.path.join(exp_dir, "evaluation")
    elif args.loss != "mse":
        exp_sub = "smoke_test_smooth_l1" if args.smoke_test else os.path.join("experiments", "smooth_l1")
        exp_dir = os.path.join(project_root, "outputs", exp_sub)
        checkpoints_dir = os.path.join(exp_dir, "checkpoints")
        plots_dir = os.path.join(exp_dir, "plots")
        history_file = os.path.join(exp_dir, "training_history.csv")
        eval_dir = os.path.join(exp_dir, "evaluation")
    else:
        checkpoints_dir = os.path.join(project_root, "outputs", "checkpoints")
        plots_dir = os.path.join(project_root, "outputs", "plots")
        history_file = os.path.join(project_root, "outputs", "smoke_test_history.csv") if args.smoke_test else os.path.join(project_root, "outputs", "training_history.csv")
        eval_dir = os.path.join(project_root, "outputs", "evaluation")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Data loaders
    train_manifest = os.path.join(project_root, "data", "splits", "train.txt")
    val_manifest = os.path.join(project_root, "data", "splits", "val.txt")
    
    train_loader = create_dataloader(train_manifest, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_manifest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    exp_name = "Baseline (MSE)" if args.loss == "mse" else "Experiment 2 — Smooth L1"
    output_dir = exp_dir if 'exp_dir' in locals() else os.path.dirname(checkpoints_dir)

    print("==================================================")
    print("TRAINING CONFIGURATION")
    print("==================================================")
    print(f"Experiment Name: {exp_name}")
    print(f"Loss Function: {args.loss.upper()}")
    print(f"Loss Configuration: {loss_meta['formulation']}")
    print(f"Random Seed: {args.seed}")
    print(f"Dataset: ADE20K")
    print(f"Training Images: {len(train_loader.dataset)}")
    print(f"Validation Images: {len(val_loader.dataset)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Optimizer: Adam")
    print(f"Learning Rate: {args.lr}")
    print(f"Scheduler: StepLR (step_size=10, gamma=0.5)")
    print(f"Epochs: {'1 (Smoke Test)' if args.smoke_test else args.epochs}")
    print(f"Device: {device} ({get_device_name()})")
    print(f"AMP: {use_amp}")
    print(f"Output Directory: {output_dir}")
    print(f"Checkpoints Directory: {checkpoints_dir}")
    print("==================================================\n")
    
    # Model
    model = ColorizationUNet().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    scaler = torch.amp.GradScaler(amp_type) if use_amp else None
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Resume Checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Verify loss consistency on resume for scientific fairness
        chk_loss = checkpoint.get("loss_name", checkpoint.get("config", {}).get("loss", "mse"))
        if chk_loss != args.loss:
            raise ValueError(
                f"[ERROR] Cannot resume experiment with loss '{args.loss}' from a checkpoint trained with loss '{chk_loss}'. "
                f"For scientific fairness, experiments with different loss functions must start from fresh random initialization."
            )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch') or 0
        
        # Check if best.pth has more accurate best_val_loss or best_epoch
        best_chk_path = os.path.join(checkpoints_dir, 'best.pth')
        if os.path.isfile(best_chk_path):
            best_chk = torch.load(best_chk_path, map_location='cpu', weights_only=False)
            best_val_loss = min(best_val_loss, best_chk.get('best_val_loss', float('inf')))
            if not best_epoch:
                best_epoch = best_chk.get('best_epoch', 0)
                
        print(f"Resumed at epoch {start_epoch + 1} (start_epoch={start_epoch}) with best_val_loss {best_val_loss:.4f} (best_epoch={best_epoch})\n")
    
    write_header = not os.path.exists(history_file) or start_epoch == 0
    
    history = {'train_loss': [], 'val_loss': []}
    if os.path.exists(history_file) and start_epoch > 0:
        with open(history_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['epoch']) <= start_epoch:
                    history['train_loss'].append(float(row['train_loss']))
                    history['val_loss'].append(float(row['val_loss']))
    
    # Smoke Test Restrictions
    num_train_batches = 5 if args.smoke_test else len(train_loader)
    num_val_batches = 2 if args.smoke_test else len(val_loader)
    num_epochs = start_epoch + 1 if args.smoke_test else args.epochs
    
    # For smoke test parameter check
    test_param = next(model.parameters())
    pre_step_param = None
    param_updated = False

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        
        # Training Loop
        for i, (l_batch, ab_batch) in enumerate(train_loader):
            if i >= num_train_batches:
                break
                
            l_batch, ab_batch = l_batch.to(device), ab_batch.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast(amp_type):
                    pred_ab = model(l_batch)
                    loss = criterion(pred_ab, ab_batch)
            else:
                pred_ab = model(l_batch)
                loss = criterion(pred_ab, ab_batch)
                
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[ERROR] Non-finite loss detected at Epoch {epoch+1}, Batch {i+1}: {loss.item()}")
                sys.exit(1)
                
            if use_amp:
                scaler.scale(loss).backward()
                if args.smoke_test and i == 0:
                    pre_step_param = test_param.clone().detach()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.smoke_test and i == 0:
                    pre_step_param = test_param.clone().detach()
                optimizer.step()
                
            if args.smoke_test and i == 0:
                if not torch.equal(test_param, pre_step_param):
                    param_updated = True
                
            running_loss += loss.item()
            
        train_loss = running_loss / num_train_batches
        
        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for i, (l_batch, ab_batch) in enumerate(val_loader):
                if i >= num_val_batches:
                    break
                l_batch, ab_batch = l_batch.to(device), ab_batch.to(device)
                if use_amp:
                    with torch.amp.autocast(amp_type):
                        pred_ab = model(l_batch)
                        loss = criterion(pred_ab, ab_batch)
                else:
                    pred_ab = model(l_batch)
                    loss = criterion(pred_ab, ab_batch)
                
                val_loss_sum += loss.item()
                
        val_loss = val_loss_sum / num_val_batches
        
        # Scheduler Step
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        mem_stats = get_memory_stats(device)
        mem_str = ""
        if "allocated_mb" in mem_stats:
            mem_str = f" | Mem: {mem_stats['allocated_mb']:.1f} MB"
            
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}{mem_str}")
              
        # Save History
        with open(history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'epoch_time'])
                write_header = False
            writer.writerow([epoch+1, train_loss, val_loss, current_lr, epoch_time])
            
        # Checkpointing
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'loss_name': args.loss,
            'loss_config': loss_meta,
            'loss_beta': args.loss_beta if args.loss == 'smooth_l1' else None,
            'config': vars(args)
        }
        
        # Checkpointing (protect baseline checkpoints during smoke test)
        save_name = "smoke_test_latest.pth" if (args.smoke_test and os.path.exists(os.path.join(checkpoints_dir, 'latest.pth'))) else "latest.pth"
        best_save_name = "smoke_test_best.pth" if (args.smoke_test and os.path.exists(os.path.join(checkpoints_dir, 'best.pth'))) else "best.pth"

        torch.save(checkpoint_state, os.path.join(checkpoints_dir, save_name))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_state['best_val_loss'] = best_val_loss
            checkpoint_state['best_epoch'] = epoch + 1
            torch.save(checkpoint_state, os.path.join(checkpoints_dir, best_save_name))
            
        # Qualitative & quantitative evaluation on fixed test set (skipped during smoke tests)
        eval_epochs = [1, 5, 10, 15, 20]
        if (epoch + 1) in eval_epochs and not args.smoke_test:
            evaluate_fixed_test_images(
                model=model,
                device=device,
                checkpoint_name=f"epoch_{epoch+1:02d}",
                epoch_num=epoch + 1,
                amp_type=amp_type,
                plots_dir=plots_dir,
                eval_dir=eval_dir
            )
            
    # Smoke test checks
    if args.smoke_test:
        print("\n==================================================")
        print("SMOKE TEST VALIDATION")
        print("==================================================")
        print(f"Loss Finite: PASS (Final Loss: {train_loss:.4f})")
        print(f"Backward Pass: PASS")
        print(f"Parameter Update Test: {'PASS' if param_updated else 'FAIL'}")
        
        # Integrity Test
        print("\nRunning Checkpoint Integrity Test...")
        fresh_model = ColorizationUNet().to(device)
        test_chk_path = os.path.join(checkpoints_dir, best_save_name) if 'best_save_name' in locals() and os.path.exists(os.path.join(checkpoints_dir, best_save_name)) else os.path.join(checkpoints_dir, 'best.pth')
        chk = torch.load(test_chk_path, map_location=device, weights_only=False)
        fresh_model.load_state_dict(chk['model_state_dict'])
        
        fresh_model.eval()
        with torch.no_grad():
            dummy_in = torch.randn(args.batch_size, 1, 256, 256).to(device)
            dummy_out = fresh_model(dummy_in)
            if list(dummy_out.shape) == [args.batch_size, 2, 256, 256]:
                print("Checkpoint Integrity Test: PASS")
            else:
                print(f"Checkpoint Integrity Test: FAIL (Shape was {list(dummy_out.shape)})")
                
        mem_stats = get_memory_stats(device)
        if "allocated_mb" in mem_stats:
            print(f"\nPeak Memory Allocated: {mem_stats['allocated_mb']:.2f} MB")
        if "reserved_mb" in mem_stats:
            print(f"Peak Memory Reserved: {mem_stats['reserved_mb']:.2f} MB")
        elif "driver_allocated_mb" in mem_stats:
            print(f"Driver Memory Allocated: {mem_stats['driver_allocated_mb']:.2f} MB")
        elif not mem_stats:
            print("\nCUDA Memory Metrics: Not applicable on this Mac")
    else:
        # Full training post-verification
        exp_name = "SMOOTH L1" if args.loss == "smooth_l1" else "BASELINE"
        print("\n==================================================")
        print(f"POST-TRAINING {exp_name} VERIFICATION & EVALUATION")
        print("==================================================")
        best_chk_path = os.path.join(checkpoints_dir, 'best.pth')
        if os.path.isfile(best_chk_path):
            best_chk = torch.load(best_chk_path, map_location=device, weights_only=False)
            best_model = ColorizationUNet().to(device)
            best_model.load_state_dict(best_chk['model_state_dict'])
            best_epoch = best_chk.get('best_epoch', best_chk.get('epoch', 0) + 1)
            best_val = best_chk.get('best_val_loss', float('inf'))
            print(f"Loaded best checkpoint from Epoch {best_epoch} (Val Loss: {best_val:.4f})")
            
            # Evaluate best checkpoint on fixed test set
            evaluate_fixed_test_images(
                model=best_model,
                device=device,
                checkpoint_name="best",
                epoch_num=best_epoch,
                amp_type=amp_type,
                plots_dir=plots_dir,
                eval_dir=eval_dir
            )
            
            # Post-training inference verification
            best_model.eval()
            with torch.no_grad():
                dummy_1 = torch.randn(1, 1, 256, 256, device=device)
                out_1 = best_model(dummy_1)
                single_pass = (list(out_1.shape) == [1, 2, 256, 256] and
                               not torch.isnan(out_1).any() and
                               not torch.isinf(out_1).any())
                print(f"Reload Inference Test (Single Image [1, 2, 256, 256]): {'PASS' if single_pass else 'FAIL'}")
                
                dummy_8 = torch.randn(8, 1, 256, 256, device=device)
                out_8 = best_model(dummy_8)
                batch_pass = (list(out_8.shape) == [8, 2, 256, 256] and
                              not torch.isnan(out_8).any() and
                              not torch.isinf(out_8).any())
                print(f"Reload Inference Test (Batch [8, 2, 256, 256]): {'PASS' if batch_pass else 'FAIL'}")
                
                weights_finite = all(torch.isfinite(p).all().item() for p in best_model.parameters())
                print(f"Model Weights Finite Check: {'PASS' if weights_finite else 'FAIL'}")
            
    # Plotting
    if len(history['train_loss']) > 0:
        plt.figure(figsize=(10, 5))
        epochs_range = list(range(1, len(history['train_loss']) + 1))
        plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        loss_ylabel = "Smooth L1 Loss" if args.loss == "smooth_l1" else "MSE Loss"
        plt.ylabel(loss_ylabel)
        plt.legend()
        plot_title = "Smooth L1 Experiment — Training & Validation Loss" if args.loss == "smooth_l1" else "Baseline — Training & Validation Loss"
        plt.title(plot_title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, 'training_loss.png'))
        plt.close()
        print(f"Loss curves saved to: {os.path.join(plots_dir, 'training_loss.png')}")

if __name__ == "__main__":
    main()
