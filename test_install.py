import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print('Testing imports...')
    try:
        from models import DeepAdaptiveAEC, LSTM_DNN, DifferentiableNLMS
        from utils.signal_processing import stft, istft
        from utils.metrics import compute_erle, compute_sdr
        from data.dataset import AECDataset
        print('  ? All imports successful')
        return True
    except Exception as e:
        print(f'  ? Import failed: {e}')
        return False


def test_model_creation():
    print('\nTesting model creation...')
    try:
        dnn_config = {
            'input_dim': 513,
            'hidden_dim': 300,
            'num_lstm_layers': 2,
            'num_fc_layers': 2,
            'dropout': 0.0
        }
        
        nlms_config = {
            'filter_length': 32,
            'step_size': 0.1,
            'eps': 1e-8
        }
        
        model = DeepAdaptiveAEC(dnn_config=dnn_config, nlms_config=nlms_config, mode='joint')
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f'  ? Model created successfully')
        print(f'  ? Total parameters: {total_params:,}')
        return True
    except Exception as e:
        print(f'  ? Model creation failed: {e}')
        return False


def test_forward_pass():
    print('\nTesting forward pass...')
    try:
        from models import DeepAdaptiveAEC
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  Using device: {device}')
        
        dnn_config = {
            'input_dim': 513,
            'hidden_dim': 300,
            'num_lstm_layers': 2,
            'num_fc_layers': 2,
            'dropout': 0.0
        }
        
        nlms_config = {
            'filter_length': 32,
            'step_size': 0.1,
            'eps': 1e-8
        }
        
        model = DeepAdaptiveAEC(dnn_config=dnn_config, nlms_config=nlms_config, mode='joint')
        model.to(device)
        
        batch_size = 2
        seq_len = 100
        freq_bins = 513
        
        Y_mag = torch.rand(batch_size, seq_len, freq_bins).to(device)
        X_mag = torch.rand(batch_size, seq_len, freq_bins).to(device)
        
        outputs = model(Y_mag, X_mag)
        
        print(f'  ? Forward pass successful')
        print(f'  ? E_hat_mag shape: {outputs["E_hat_mag"].shape}')
        print(f'  ? X_hat_mag shape: {outputs["X_hat_mag"].shape}')
        return True
    except Exception as e:
        print(f'  ? Forward pass failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    print('\nTesting dataset creation...')
    try:
        from data.dataset import AECDataset
        
        dataset = AECDataset(
            num_samples=5,
            sample_rate=16000,
            duration=1,
            n_fft=1024,
            hop_length=256
        )
        
        sample = dataset[0]
        
        print(f'  ? Dataset created successfully')
        print(f'  ? Y_mag shape: {sample["Y_mag"].shape}')
        print(f'  ? X_mag shape: {sample["X_mag"].shape}')
        print(f'  ? T_mag shape: {sample["T_mag"].shape}')
        return True
    except Exception as e:
        print(f'  ? Dataset test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    print('=' * 60)
    print('Deep Adaptive AEC - Test Suite')
    print('=' * 60)
    
    results = []
    results.append(('Imports', test_imports()))
    results.append(('Model Creation', test_model_creation()))
    results.append(('Forward Pass', test_forward_pass()))
    results.append(('Dataset', test_dataset()))
    
    print('\n' + '=' * 60)
    print('Test Summary:')
    print('=' * 60)
    all_passed = True
    for name, passed in results:
        status = '? PASSED' if passed else '? FAILED'
        print(f'  {name:20s} {status}')
        if not passed:
            all_passed = False
    
    print('=' * 60)
    if all_passed:
        print('All tests passed! Ready to train.')
    else:
        print('Some tests failed. Please check the errors above.')
    print('=' * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())