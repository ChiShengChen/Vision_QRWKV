import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms # torchvision.datasets is not used directly for MedMNIST
# MedMNIST import
from medmnist import PathMNIST # Changed for PathMNIST
from quantum_rwkv import ModelConfig, QuantumRWKVModel
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import datetime

# Function name changed for PathMNIST
def run_quantum_pathmnist_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Updated print statement for PathMNIST
    print(f"Using device: {device} for quantum PathMNIST classification test")

    training_start_time_obj = datetime.datetime.now()
    training_start_time_str = training_start_time_obj.strftime("%Y%m%d_%H%M%S")

    # PathMNIST Data loading and preprocessing
    image_size = 8
    batch_size_train = 64
    batch_size_test = 1000

    # Specifics for PathMNIST
    dataset_name_lower = "pathmnist"
    input_channels = 3
    num_classes = 9
    # Normalization for PathMNIST (RGB)
    mean_val = (0.7024, 0.5462, 0.6964)
    std_val = (0.2388, 0.2820, 0.2157)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # Converts to (C, H, W) and scales to [0,1]
        transforms.Normalize(mean=mean_val, std=std_val)
    ])
    
    # Create a unique results directory for this run
    # Updated base directory name for PathMNIST
    base_results_dir = f"results_{dataset_name_lower}_quantum"
    results_dir = f"{base_results_dir}_{training_start_time_str}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # PathMNIST DataLoaders
    train_dataset = PathMNIST(split='train', transform=transform, download=True, as_rgb=True, root='./data')
    test_dataset = PathMNIST(split='test', transform=transform, download=True, as_rgb=True, root='./data')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test, shuffle=False)

    # Model Configuration
    seq_len = image_size * image_size
    n_embd_test = 64
    n_head_test = 4
    n_layer_test = 2
    # input_dim_test updated to number of channels
    input_dim_test = input_channels
    # output_dim_test updated to number of classes for PathMNIST
    output_dim_test = num_classes
    n_qubits_test = 8 # Kept as per original quantum config
    q_depth_test = 1  # Kept as per original quantum config

    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        block_size=seq_len + 10,
        n_intermediate=n_embd_test * 4,
        layer_norm_epsilon=1e-5,
        input_dim=input_dim_test,
        output_dim=output_dim_test,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
    )
    # Updated print statement for PathMNIST config
    print(f"Quantum Model Config for PathMNIST: {config}\n")

    try:
        model = QuantumRWKVModel(config)
        model.to(device)
    except Exception as e:
        print(f"Error instantiating quantum RWKVModel: {e}")
        raise
    # Updated print statement
    print(f"Quantum RWKVModel for PathMNIST instantiated successfully.\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30
    print_every = 1

    all_epoch_train_losses = []
    all_epoch_test_accuracies = []

    # Prepare CSV for epoch-wise metrics
    # Updated CSV filename
    epoch_metrics_csv_filename = os.path.join(results_dir, f"epoch_metrics_quantum_{dataset_name_lower}.csv")
    epoch_metrics_header = ["Epoch", "Average_Training_Loss", "Test_Accuracy_Percent"]
    try:
        with open(epoch_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_metrics_header)
    except Exception as e:
        print(f"Error creating epoch metrics CSV {epoch_metrics_csv_filename}: {e}")

    # Updated print statement
    print(f"Starting quantum training for PathMNIST classification...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_processed = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Reshape for RWKV: (batch, seq_len, input_dim)
            data = data.permute(0, 2, 3, 1).contiguous().view(data.size(0), -1, input_channels).to(device)
            target = target.squeeze().long().to(device) # Ensure target is 1D LongTensor
            
            optimizer.zero_grad()
            initial_states = None
            predictions, _ = model(data, states=initial_states)
            output_for_loss = predictions[:, -1, :]
            loss = criterion(output_for_loss, target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            num_batches_processed += 1
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        average_epoch_train_loss = epoch_loss_sum / num_batches_processed
        all_epoch_train_losses.append(average_epoch_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {average_epoch_train_loss:.6f}")

        model.eval()
        test_loss_sum = 0
        correct_preds = 0
        all_preds_epoch = []
        all_targets_epoch = []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.permute(0, 2, 3, 1).contiguous().view(data.size(0), -1, input_channels).to(device)
                target = target.squeeze().long().to(device)
                initial_states = None
                predictions, _ = model(data, states=initial_states)
                output_for_loss = predictions[:, -1, :]
                test_loss_sum += criterion(output_for_loss, target).item() * data.size(0)
                pred = output_for_loss.argmax(dim=1, keepdim=True)
                correct_preds += pred.eq(target.view_as(pred)).sum().item()
                all_preds_epoch.extend(pred.cpu().numpy().flatten())
                all_targets_epoch.extend(target.cpu().numpy().flatten())

        average_test_loss = test_loss_sum / len(test_loader.dataset)
        current_epoch_accuracy = 100. * correct_preds / len(test_loader.dataset)
        all_epoch_test_accuracies.append(current_epoch_accuracy)
        print(f'Test set: Average loss: {average_test_loss:.4f}, Accuracy: {correct_preds}/{len(test_loader.dataset)} ({current_epoch_accuracy:.2f}%)\n')

        try:
            with open(epoch_metrics_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, f"{average_epoch_train_loss:.6f}", f"{current_epoch_accuracy:.2f}"])
        except Exception as e:
            print(f"Error writing to epoch metrics CSV {epoch_metrics_csv_filename} for epoch {epoch+1}: {e}")

        if (epoch + 1) % (print_every * 5) == 0 or epoch == num_epochs -1 :
            cm = confusion_matrix(all_targets_epoch, all_preds_epoch, labels=list(range(num_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                        xticklabels=range(num_classes), yticklabels=range(num_classes))
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            # Updated plot title
            plt.title(f'Confusion Matrix - Epoch {epoch+1} (Quantum PathMNIST)')
            # Updated plot filename
            cm_plot_filename = os.path.join(results_dir, f"confusion_matrix_quantum_{dataset_name_lower}_epoch{epoch+1}_{training_start_time_str}.png")
            try:
                plt.savefig(cm_plot_filename)
                print(f"Confusion matrix plot saved as {cm_plot_filename}")
                plt.close()
            except Exception as e:
                print(f"Error saving confusion matrix plot: {e}")

    # Updated print statement
    print(f"Quantum PathMNIST training finished.\n")
    final_accuracy = all_epoch_test_accuracies[-1] if all_epoch_test_accuracies else 0
    final_train_loss = all_epoch_train_losses[-1] if all_epoch_train_losses else float('inf')

    # Updated summary CSV filename
    overall_metrics_csv_filename = os.path.join(results_dir, f"model_performance_summary_quantum_{dataset_name_lower}.csv")
    header = [
        'Timestamp_Run_Start', 'Experiment_ID', 'Model_Type', 'Task',
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth',
        'learning_rate', 'num_epochs_run', 'seq_len_data', 'input_channels_data',
        'Config_Block_Size', 'Config_n_intermediate', 'num_classes_data',
        'Final_Test_Accuracy_Percent', 'Final_Avg_Train_Loss'
    ]
    # Updated experiment_id for PathMNIST
    experiment_id = f"q_{dataset_name_lower}_{training_start_time_str}"
    learning_rate = optimizer.param_groups[0]['lr']
    data_row = [
        training_start_time_obj.strftime("%Y-%m-%d_%H-%M-%S"),
        experiment_id, 'Quantum', f'PathMNIST_8x8', # Updated Task name
        config.n_layer, config.n_embd, config.n_head, config.n_qubits, config.q_depth,
        f'{learning_rate:.1e}', num_epochs, seq_len, input_channels,
        config.block_size, config.n_intermediate, num_classes,
        f'{final_accuracy:.2f}', f'{final_train_loss:.6f}'
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
    if all_epoch_train_losses:
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), all_epoch_train_losses, label='Average Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # Updated plot title
        plt.title('Training Loss Over Epochs (Quantum PathMNIST)')
        plt.legend()
        plt.grid(True)

    if all_epoch_test_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), all_epoch_test_accuracies, label='Test Accuracy', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        # Updated plot title
        plt.title('Test Accuracy Over Epochs (Quantum PathMNIST)')
        plt.legend()
        plt.grid(True)

    if all_epoch_train_losses or all_epoch_test_accuracies:
        plt.tight_layout()
        # Updated plot filename
        epoch_plot_filename = os.path.join(results_dir, f"training_progress_quantum_{dataset_name_lower}_{training_start_time_str}.png")
        try:
            plt.savefig(epoch_plot_filename)
            print(f"Training progress plot saved as {epoch_plot_filename}")
            plt.close('all')
        except Exception as e:
            print(f"Error saving training progress plot: {e}")
    else:
        plt.close('all')

    # Updated final print statement
    print(f"\n=== Finished Quantum PathMNIST Classification Test ===\n")

if __name__ == '__main__':
    # Updated function call
    run_quantum_pathmnist_classification() 