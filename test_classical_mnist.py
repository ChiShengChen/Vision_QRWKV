import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from rwkv import ModelConfig, RWKVModel
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import datetime

def run_classical_mnist_classification():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for classical MNIST classification test")

    training_start_time_obj = datetime.datetime.now()
    training_start_time_str = training_start_time_obj.strftime("%Y%m%d_%H%M%S")
    
    # MNIST Data loading and preprocessing
    image_size = 8
    batch_size_train = 64
    batch_size_test = 1000

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])

    # Create a unique results directory for this run
    base_results_dir = "results_mnist_classical"
    results_dir = f"{base_results_dir}_{training_start_time_str}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # MNIST DataLoaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size_test, shuffle=False)

    # Model Configuration
    seq_len = image_size * image_size  # 8x8 image flattened
    n_embd_test = 64  # Embedding dimension
    n_head_test = 4
    n_layer_test = 2
    input_dim_test = 1  # Each pixel is a single value
    output_dim_test = 10  # 10 classes for MNIST digits

    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        block_size=seq_len + 10, # Sequence length is 8*8 = 64
        n_intermediate=n_embd_test * 4, # Adjusted for complexity
        layer_norm_epsilon=1e-5,
        input_dim=input_dim_test,
        output_dim=output_dim_test
    )
    print(f"Classical Model Config for MNIST: {config}\n")

    try:
        model = RWKVModel(config)
        model.to(device)
    except Exception as e:
        print(f"Error instantiating classical RWKVModel: {e}")
        raise
    print("Classical RWKVModel for MNIST instantiated successfully.\n")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # Use CrossEntropyLoss for classification
    num_epochs = 30 # Reduced for quicker testing
    print_every = 1

    all_epoch_train_losses = [] # Renamed for clarity
    all_epoch_test_accuracies = [] # Renamed for clarity

    # Prepare CSV for epoch-wise metrics
    epoch_metrics_csv_filename = os.path.join(results_dir, "epoch_metrics_classical.csv")
    epoch_metrics_header = ["Epoch", "Average_Training_Loss", "Test_Accuracy_Percent"]
    try:
        with open(epoch_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(epoch_metrics_header)
    except Exception as e:
        print(f"Error creating epoch metrics CSV {epoch_metrics_csv_filename}: {e}")

    print("Starting classical training for MNIST classification...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0 # Sum of losses in an epoch
        num_batches_processed = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1, 1).to(device)  # Flatten image and add channel dim
            target = target.to(device)

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

        # Evaluation on test set
        model.eval()
        test_loss_sum = 0 # Sum of test losses
        correct_preds = 0 # Total correct predictions
        all_preds_epoch = []
        all_targets_epoch = []
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1, 1).to(device)
                target = target.to(device)
                initial_states = None
                predictions, _ = model(data, states=initial_states)
                output_for_loss = predictions[:, -1, :]
                test_loss_sum += criterion(output_for_loss, target).item() * data.size(0) # Weighted sum
                pred = output_for_loss.argmax(dim=1, keepdim=True)
                correct_preds += pred.eq(target.view_as(pred)).sum().item()
                all_preds_epoch.extend(pred.cpu().numpy())
                all_targets_epoch.extend(target.cpu().numpy())

        average_test_loss = test_loss_sum / len(test_loader.dataset)
        current_epoch_accuracy = 100. * correct_preds / len(test_loader.dataset)
        all_epoch_test_accuracies.append(current_epoch_accuracy)

        print(f'Test set: Average loss: {average_test_loss:.4f}, Accuracy: {correct_preds}/{len(test_loader.dataset)} ({current_epoch_accuracy:.2f}%)\n')

        # Append current epoch's metrics to CSV
        try:
            with open(epoch_metrics_csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, f"{average_epoch_train_loss:.6f}", f"{current_epoch_accuracy:.2f}"])
        except Exception as e:
            print(f"Error writing to epoch metrics CSV {epoch_metrics_csv_filename} for epoch {epoch+1}: {e}")

        # Plot confusion matrix
        if (epoch + 1) % (print_every * 5) == 0 or epoch == num_epochs -1 :
            cm = confusion_matrix(all_targets_epoch, all_preds_epoch)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - Epoch {epoch+1} (Classical)')
            cm_plot_filename = os.path.join(results_dir, f"confusion_matrix_classical_epoch{epoch+1}.png")
            try:
                plt.savefig(cm_plot_filename)
                print(f"Confusion matrix plot saved as {cm_plot_filename}")
                plt.close()
            except Exception as e:
                print(f"Error saving confusion matrix plot: {e}")

    print("Classical training finished.\n")

    final_accuracy = all_epoch_test_accuracies[-1]
    final_train_loss = all_epoch_train_losses[-1]

    # Save overall model performance metrics
    # This CSV is now saved inside the timestamped results_dir
    overall_metrics_csv_filename = os.path.join(results_dir, "model_performance_summary_classical.csv")
    header = [
        'Timestamp_Run_Start', 'Experiment_ID', 'Model_Type', 'Task',
        'n_layer', 'n_embd', 'n_head', 'n_qubits', 'q_depth',
        'learning_rate', 'num_epochs_run', 'seq_len_train',
        'Config_Block_Size', 'Config_n_intermediate',
        'Final_Test_Accuracy_Percent', 'Final_Avg_Train_Loss'
    ]
    # experiment_id uses the same training_start_time_str for consistency within the folder
    experiment_id = f"c_mnist_{training_start_time_str}"
    learning_rate = optimizer.param_groups[0]['lr']

    data_row = [
        training_start_time_obj.strftime("%Y-%m-%d_%H-%M-%S"), # More readable timestamp for summary
        experiment_id, 'Classical', 'MNIST_8x8',
        config.n_layer, config.n_embd, config.n_head, 'N/A', 'N/A',
        f'{learning_rate:.1e}', num_epochs, seq_len,
        config.block_size, config.n_intermediate,
        f'{final_accuracy:.2f}', f'{final_train_loss:.6f}' # Accuracy already in percent
    ]

    # Check if the summary CSV exists in the main classical results folder, create/append there.
    # OR save it inside the specific run folder. Let's save it inside the specific run folder.
    # file_exists = os.path.isfile(overall_metrics_csv_filename)
    # is_empty = os.path.getsize(overall_metrics_csv_filename) == 0 if file_exists else True
    try:
        with open(overall_metrics_csv_filename, 'w', newline='', encoding='utf-8') as csvfile: # 'w' to create new for each run
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(data_row)
        print(f"Overall model performance summary saved to {overall_metrics_csv_filename}")
    except Exception as e:
        print(f"Error writing overall model performance to CSV {overall_metrics_csv_filename}: {e}")

    # Plot training loss and accuracy over epochs
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), all_epoch_train_losses, label='Average Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs (Classical)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), all_epoch_test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Over Epochs (Classical)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Save plot to the new timestamped results_dir
    epoch_plot_filename = os.path.join(results_dir, f"training_progress_classical_{training_start_time_str}.png")
    try:
        plt.savefig(epoch_plot_filename)
        print(f"Training progress plot saved as {epoch_plot_filename}")
        plt.close('all') # Close all figures
    except Exception as e:
        print(f"Error saving training progress plot: {e}")

    # Removed the separate epoch_losses_classical.csv as it's now combined in epoch_metrics_classical.csv

    print("\n=== Finished Classical MNIST Classification Test ===\n")

if __name__ == '__main__':
    run_classical_mnist_classification() 