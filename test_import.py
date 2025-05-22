"""Test script to verify NES Copilot installation."""

import sys
import io

# Set stdout to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import nes_copilot
    from nes_copilot import config_manager, data_manager, logging_manager
    
    print("[OK] Successfully imported nes_copilot and core modules!")
    print(f"NES Copilot version: {getattr(nes_copilot, '__version__', '0.1.0')}")
    
    # Load configuration from file
    config_path = 'test_config.yaml'
    config_manager = config_manager.ConfigManager(config_path)
    print("[OK] Successfully created ConfigManager")
    
    data_manager = data_manager.DataManager(config_manager)
    print("[OK] Successfully created DataManager")
    
    logging_manager = logging_manager.LoggingManager(data_manager)
    print("[OK] Successfully created LoggingManager")
    
    print("\n[SUCCESS] All tests passed! NES Copilot is working correctly.")
    
except Exception as e:
    print(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
