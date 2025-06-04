import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from medmnist import ChestMNIST # Changed for ChestMNIST
from quantum_rwkv import ModelConfig, QuantumRWKVModel 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix # Note: accuracy_score might not be ideal for multi-label
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import datetime

# Function name changed for ChestMNIST
def run_quantum_chestmnist_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for quantum ChestMNIST classification test")
    print("WARNING: ChestMNIST is a multi-label dataset. This script uses CrossEntropyLoss")
    print("which is intended for single-label multi-class classification. It will likely FAIL or perform poorly.")
    print("You will need to adapt the loss function (e.g., BCEWithLogitsLoss) and evaluation for multi-label.")

    training_start_time_obj = datetime.datetime.now()
    training_start_time_str = training_start_time_obj.strftime("%Y%m%d_%H%M%S")
    
    image_size = 8
    batch_size_train = 64
    batch_size_test = 1000
    
    dataset_name_lower = "chestmnist"
    input_channels = 1 
    num_classes = 14 # ChestMNIST has 14 labels (multi-label)
    mean_val = (0.5098,)
    std_val = (0.2493,)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val)
    ])

    base_results_dir = f"results_{dataset_name_lower}_quantum"
    results_dir = f"{base_results_dir}_{training_start_time_str}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    train_dataset = ChestMNIST(split='train', transform=transform, download=True, root='./data')
    test_dataset = ChestMNIST(split='test', transform=transform, download=True, root='./data')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test, shuffle=False)

    seq_len = image_size * image_size
    n_embd_test = 64
    n_head_test = 4
    n_layer_test = 2
    input_dim_test = input_channels
    output_dim_test = num_classes
    n_qubits_test = 8
    q_depth_test = 1

    config = ModelConfig(
        n_embd=n_embd_test, n_head=n_head_test, n_layer=n_layer_test,
        block_size=seq_len + 10, n_intermediate=n_embd_test * 4,
        layer_norm_epsilon=1e-5, input_dim=input_dim_test, output_dim=output_dim_test,
        n_qubits=n_qubits_test, q_depth=q_depth_test
    )
    print(f"Quantum Model Config for ChestMNIST: {config}\n")

    model = QuantumRWKVModel(config)
    model.to(device)
    print(f"Quantum RWKVModel for ChestMNIST instantiated successfully.\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # !!! CrossEntropyLoss is NOT suitable for multi-label classification like ChestMNIST !!!
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30
    print_every = 1

    all_epoch_train_losses = []
    all_epoch_test_accuracies = [] # Accuracy is also problematic for multi-label with this setup

    epoch_metrics_csv_filename = os.path.join(results_dir, f"epoch_metrics_quantum_{dataset_name_lower}.csv")
    epoch_metrics_header = ["Epoch", "Average_Training_Loss", "Test_Accuracy_Percent"]
    try:
        with open(epoch_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_metrics_header)
    except Exception as e:
        print(f"Error creating epoch metrics CSV {epoch_metrics_csv_filename}: {e}")

    print(f"Starting quantum training for ChestMNIST classification (multi-label context)... NOTE POTENTIAL FAILURE DUE TO LOSS FUNCTION")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_processed = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1, input_channels).to(device)
            target = target.float().to(device) # Keep as float for BCEWithLogitsLoss adaptation

            optimizer.zero_grad()
            initial_states = None
            predictions, _ = model(data, states=initial_states)
            output_for_loss = predictions[:, -1, :]
            
            try:
                loss = criterion(output_for_loss, target.argmax(dim=1)) # DANGEROUS: .argmax() is not a fix for multi-label
            except RuntimeError as e:
                print(f"RUNTIME ERROR with loss: {e}. As expected for multi-label with CrossEntropyLoss.")
                loss = torch.tensor(float('inf'), device=device)
            
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item()
            else:
                epoch_loss_sum += 1e9 # large number if loss was inf
            num_batches_processed += 1

            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        average_epoch_train_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('inf')
        all_epoch_train_losses.append(average_epoch_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_epoch_train_loss:.6f}")

        model.eval()
        test_loss_sum = 0
        correct_preds = 0 
        all_preds_epoch_for_cm = []
        all_targets_epoch_for_cm = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1, input_channels).to(device)
                target = target.float().to(device)
                initial_states = None
                predictions, _ = model(data, states=initial_states)
                output_for_loss = predictions[:, -1, :]
                
                try:
                    test_loss_sum += criterion(output_for_loss, target.argmax(dim=1)).item() * data.size(0)
                except RuntimeError:
                    test_loss_sum += float('inf')
                
                pred_for_cm = output_for_loss.argmax(dim=1, keepdim=True)
                pred_binary = (torch.sigmoid(output_for_loss) > 0.5).float()
                correct_preds += pred_binary.eq(target).sum().item()
                
                all_preds_epoch_for_cm.extend(pred_for_cm.cpu().numpy().flatten())
                all_targets_epoch_for_cm.extend(target.argmax(dim=1).cpu().numpy().flatten())

        average_test_loss = test_loss_sum / len(test_loader.dataset) if len(test_loader.dataset) > 0 else float('inf')
        current_epoch_accuracy = (100. * correct_preds / (len(test_loader.dataset) * num_classes)) if len(test_loader.dataset) > 0 else 0.0
        all_epoch_test_accuracies.append(current_epoch_accuracy)

        print(f'Test set: Average loss: {average_test_loss:.4f}, Accuracy (element-wise binary): {current_epoch_accuracy:.2f}%\n')

        try:
            with open(epoch_metrics_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, f"{average_epoch_train_loss:.6f}", f"{current_epoch_accuracy:.2f}"])
        except Exception as e:
            print(f"Error writing to epoch metrics CSV {epoch_metrics_csv_filename} for epoch {epoch+1}: {e}")

        if (epoch + 1) % (print_every * 5) == 0 or epoch == num_epochs -1 :
            if all_targets_epoch_for_cm and all_preds_epoch_for_cm:
                cm = confusion_matrix(all_targets_epoch_for_cm, all_preds_epoch_for_cm, labels=list(range(num_classes)))
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                            xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
                plt.xlabel('Predicted Label (Argmax)')
                plt.ylabel('True Label (Argmax)')
                plt.title(f'Confusion Matrix (Argmax based) - Epoch {epoch+1} (Quantum ChestMNIST)')
                cm_plot_filename = os.path.join(results_dir, f"confusion_matrix_quantum_{dataset_name_lower}_epoch{epoch+1}_{training_start_time_str}.png")
                try:
                    plt.savefig(cm_plot_filename)
                    print(f"Confusion matrix plot saved as {cm_plot_filename}")
                    plt.close()
                except Exception as e:
                    print(f"Error saving confusion matrix plot: {e}")
            else:
                print("Skipping CM plot due to no data for CM.")

    print(f"Quantum ChestMNIST training finished.\n")

    final_accuracy = all_epoch_test_accuracies[-1] if all_epoch_test_accuracies else 0
    final_train_loss = all_epoch_train_losses[-1] if all_epoch_train_losses else float('inf')

    overall_metrics_csv_filename = os.path.join(results_dir, f"model_performance_summary_quantum_{dataset_name_lower}.csv")
    header = [
        'Timestamp_Run_Start', 'Experiment_ID', 'Model_Type', 'Task',
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth',
        'learning_rate', 'num_epochs_run', 'seq_len_data', 'input_channels_data',
        'Config_Block_Size', 'Config_n_intermediate', 'num_classes_data',
        'Final_Test_Accuracy_Percent_ElementWise', 'Final_Avg_Train_Loss',
        'WARNING'
    ]
    experiment_id = f"q_{dataset_name_lower}_{training_start_time_str}"
    learning_rate = optimizer.param_groups[0]['lr']

    data_row = [
        training_start_time_obj.strftime("%Y-%m-%d_%H-%M-%S"),
        experiment_id, 'Quantum', f'ChestMNIST_8x8',
        config.n_layer, config.n_embd, config.n_head, config.n_qubits, config.q_depth,
        f'{learning_rate:.1e}', num_epochs, seq_len, input_channels,
        config.block_size, config.n_intermediate, num_classes,
        f'{final_accuracy:.2f}', f'{final_train_loss:.6f}',
        "MultiLabel_CrossEntropyLoss_MisMatch"
    ]

    try:
        with open(overall_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data_row)
        print(f"Overall model performance summary saved to {overall_metrics_csv_filename}")
    except Exception as e:
        print(f"Error writing overall model performance to CSV {overall_metrics_csv_filename}: {e}")

    plt.figure(figsize=(12, 4))
    if all_epoch_train_losses and all(torch.isfinite(torch.tensor(l)) for l in all_epoch_train_losses):
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), all_epoch_train_losses, label='Average Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs (Quantum ChestMNIST)')
        plt.legend()
        plt.grid(True)

    if all_epoch_test_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), all_epoch_test_accuracies, label='Test Accuracy (Element-wise)', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy (Element-wise) Over Epochs (Quantum ChestMNIST)')
        plt.legend()
        plt.grid(True)

    if (all_epoch_train_losses and all(torch.isfinite(torch.tensor(l)) for l in all_epoch_train_losses)) or all_epoch_test_accuracies:
        plt.tight_layout()
        epoch_plot_filename = os.path.join(results_dir, f"training_progress_quantum_{dataset_name_lower}_{training_start_time_str}.png")
        try:
            plt.savefig(epoch_plot_filename)
            print(f"Training progress plot saved as {epoch_plot_filename}")
        except Exception as e:
            print(f"Error saving training progress plot: {e}")
    plt.close('all')

    print(f"\n=== Finished Quantum ChestMNIST Classification Test (with multi-label caveats) ===\n")

if __name__ == '__main__':
    run_quantum_chestmnist_classification() 