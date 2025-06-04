import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from medmnist import BreastMNIST # Changed for BreastMNIST
from quantum_rwkv import ModelConfig, QuantumRWKVModel
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import datetime

def run_quantum_breastmnist_classification(): # Renamed function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for quantum BreastMNIST classification test") # Updated print

    training_start_time_obj = datetime.datetime.now()
    training_start_time_str = training_start_time_obj.strftime("%Y%m%d_%H%M%S")
    
    image_size = 8
    batch_size_train = 64
    batch_size_test = 1000
    
    dataset_name_lower = "breastmnist"
    input_channels = 1 
    num_classes = 2 # Binary classification
    mean_val = (0.2124,)
    std_val = (0.1709,)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val)
    ])

    base_results_dir = f"results_{dataset_name_lower}_quantum"
    results_dir = f"{base_results_dir}_{training_start_time_str}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    train_dataset = BreastMNIST(split='train', transform=transform, download=True, root='./data')
    test_dataset = BreastMNIST(split='test', transform=transform, download=True, root='./data')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    seq_len = image_size * image_size
    n_qubits_test = 8
    q_depth_test = 1
    config = ModelConfig(
        n_embd=64, n_head=4, n_layer=2,
        block_size=seq_len + 10, n_intermediate=64 * 4,
        layer_norm_epsilon=1e-5, input_dim=input_channels, output_dim=num_classes,
        n_qubits=n_qubits_test, q_depth=q_depth_test
    )
    print(f"Quantum Model Config for BreastMNIST: {config}\n")

    model = QuantumRWKVModel(config)
    model.to(device)
    print(f"Quantum RWKVModel for BreastMNIST instantiated successfully.\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30
    print_every = 1

    all_epoch_train_losses = []
    all_epoch_test_accuracies = []

    epoch_metrics_csv_filename = os.path.join(results_dir, f"epoch_metrics_quantum_{dataset_name_lower}.csv")
    epoch_metrics_header = ["Epoch", "Average_Training_Loss", "Test_Accuracy_Percent"]
    try:
        with open(epoch_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_metrics_header)
    except Exception as e:
        print(f"Error creating epoch metrics CSV {epoch_metrics_csv_filename}: {e}")

    print(f"Starting quantum training for BreastMNIST classification...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        num_batches_processed = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1, input_channels).to(device)
            target = target.squeeze().long().to(device)

            optimizer.zero_grad()
            predictions, _ = model(data, states=None)
            loss = criterion(predictions[:, -1, :], target)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            num_batches_processed += 1
            if batch_idx % 100 == 0:
                print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

        average_epoch_train_loss = epoch_loss_sum / num_batches_processed
        all_epoch_train_losses.append(average_epoch_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Training Loss: {average_epoch_train_loss:.6f}")

        model.eval()
        correct_preds = 0
        all_preds_epoch, all_targets_epoch = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1, input_channels).to(device)
                target_squeezed = target.squeeze().long().to(device)
                predictions, _ = model(data, states=None)
                pred = predictions[:, -1, :].argmax(dim=1, keepdim=True)
                correct_preds += pred.eq(target_squeezed.view_as(pred)).sum().item()
                all_preds_epoch.extend(pred.cpu().numpy().flatten())
                all_targets_epoch.extend(target_squeezed.cpu().numpy().flatten())
        
        current_epoch_accuracy = 100. * correct_preds / len(test_loader.dataset)
        all_epoch_test_accuracies.append(current_epoch_accuracy)
        print(f'Test set: Accuracy: {correct_preds}/{len(test_loader.dataset)} ({current_epoch_accuracy:.2f}%)\n')

        try:
            with open(epoch_metrics_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, f"{average_epoch_train_loss:.6f}", f"{current_epoch_accuracy:.2f}"])
        except Exception as e:
            print(f"Error writing to epoch metrics CSV {epoch_metrics_csv_filename} for epoch {epoch+1}: {e}")

        if (epoch + 1) % (print_every * 5) == 0 or epoch == num_epochs -1 :
            cm = confusion_matrix(all_targets_epoch, all_preds_epoch, labels=list(range(num_classes)))
            plt.figure(figsize=(6, 4)) # Adjusted for binary
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=range(num_classes), yticklabels=range(num_classes))
            plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'CM Epoch {epoch+1} (Quantum BreastMNIST)')
            cm_plot_fn = os.path.join(results_dir, f"cm_quantum_{dataset_name_lower}_epoch{epoch+1}_{training_start_time_str}.png")
            try: plt.savefig(cm_plot_fn); print(f"CM plot saved: {cm_plot_fn}"); plt.close()
            except Exception as e: print(f"Error saving CM plot: {e}")

    print(f"Quantum BreastMNIST training finished.\n")
    final_accuracy = all_epoch_test_accuracies[-1] if all_epoch_test_accuracies else 0
    final_train_loss = all_epoch_train_losses[-1] if all_epoch_train_losses else float('inf')

    overall_metrics_csv_filename = os.path.join(results_dir, f"summary_quantum_{dataset_name_lower}.csv")
    header = [
        'Timestamp_Run_Start', 'Experiment_ID', 'Model_Type', 'Task',
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth',
        'learning_rate', 'num_epochs_run', 'seq_len_data', 'input_channels_data',
        'Config_Block_Size', 'Config_n_intermediate', 'num_classes_data',
        'Final_Test_Accuracy_Percent', 'Final_Avg_Train_Loss'
    ]
    experiment_id = f"q_{dataset_name_lower}_{training_start_time_str}"
    data_row = [
        training_start_time_obj.strftime("%Y-%m-%d_%H-%M-%S"), experiment_id, 'Quantum', f'{dataset_name_lower.capitalize()}MNIST_8x8',
        config.n_layer, config.n_embd, config.n_head, config.n_qubits, config.q_depth, 
        optimizer.param_groups[0]['lr'], num_epochs, seq_len, input_channels,
        config.block_size, config.n_intermediate, num_classes, 
        f'{final_accuracy:.2f}', f'{final_train_loss:.6f}'
    ]
    try:
        with open(overall_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data_row)
        print(f"Overall summary saved to {overall_metrics_csv_filename}")
    except Exception as e: print(f"Error writing summary CSV: {e}")

    plt.figure(figsize=(12,4)); plt.subplot(1,2,1); plt.plot(all_epoch_train_losses, label='Train Loss'); plt.title('Train Loss (Quantum BreastMNIST)')
    plt.subplot(1,2,2); plt.plot(all_epoch_test_accuracies, label='Test Acc', color='purple'); plt.title('Test Acc (Quantum BreastMNIST)'); plt.tight_layout()
    plot_fn = os.path.join(results_dir, f"progress_quantum_{dataset_name_lower}_{training_start_time_str}.png")
    try: plt.savefig(plot_fn); print(f"Progress plot saved: {plot_fn}"); plt.close('all')
    except Exception as e: print(f"Error saving progress plot: {e}")
    print(f"\n=== Finished Quantum BreastMNIST Classification Test ===\n")

if __name__ == '__main__':
    run_quantum_breastmnist_classification() 