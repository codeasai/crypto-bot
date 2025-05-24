import os
import sys
from typing import Callable, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow imports from other directories
current_file_dir = Path(__file__).parent
project_root = current_file_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary components from the existing training script
# We expect train_dqn_agent to be adapted to accept progress_callback, use_gpu_flag, and run_config_params
# and to have its internal input() prompts removed or bypassed.
from scripts.train_agent import (
    train_dqn_agent,
    load_or_collect_data,
    # We might not need all of these if train_dqn_agent handles them internally
    # and if Streamlit passes pre-validated data/configurations.
    # For now, focus on calling train_dqn_agent.
)
from utils.logger import setup_logger # Assuming this is shareable or we use Streamlit's logging

logger = setup_logger('streamlit_bridge') # Or use a passed-in callback for all logging

# Global variable to handle cancellation from Streamlit
# This needs to be accessible by the adapted train_dqn_agent
# One way is for train_dqn_agent to check this module's variable.
# Alternatively, train_dqn_agent could take a cancellation_flag object.
# For now, train_agent.py has its own 'training_cancelled' global.
# We need a way for Streamlit to set that. This bridge can expose a function.
# Or, the callback could return a "should_cancel" boolean.

# Placeholder: This assumes train_agent.py will be modified to check this or a passed-in flag.
# This is a simplification. A more robust way would be to pass a threading.Event or similar.
_streamlit_training_cancelled = False

def set_streamlit_training_cancelled(cancel_flag: bool):
    global _streamlit_training_cancelled
    # This is tricky because train_agent.py has its own 'training_cancelled'
    # This will require train_agent.py to check this flag from this module,
    # or for train_dqn_agent to accept a cancellation callback/object.
    # For now, this function is a placeholder for the concept.
    # The actual cancellation mechanism will depend on how train_agent.py is refactored.
    logger.info(f"Streamlit bridge: set_streamlit_training_cancelled called with {cancel_flag}")
    # Ideally: scripts.train_agent.training_cancelled = cancel_flag

def start_training_for_streamlit(
    symbol: str,
    timeframe: str,
    start_date_str: str,
    end_date_str: str,
    initial_balance: float,
    window_size: int,
    batch_size: int,
    episodes: int,
    output_dir_str: str,
    use_gpu_flag: bool,
    progress_callback: Callable[[Dict[str, Any]], None],
    # data_validation_callback: Callable[[pd.DataFrame], bool] # For UI-based data validation
):
    """
    Wrapper function to initiate model training from Streamlit.

    Args:
        progress_callback: Function to send updates to Streamlit UI.
                           Expected dict format:
                           {"type": "log", "level": "info/warn/error", "message": "..."}
                           {"type": "progress", "current_episode": X, "total_episodes": Y, "metrics": {...}}
                           {"type": "status", "message": "..."}
                           {"type": "result", "success": True/False, "path": "model_path_if_success", "error_message": "..."}
        # data_validation_callback: Callback to ask Streamlit UI to confirm data.
    """
    global _streamlit_training_cancelled
    _streamlit_training_cancelled = False # Reset cancel flag

    try:
        progress_callback({"type": "status", "message": "Training process started by Streamlit bridge."})

        # Parameters for DQNAgent.save() via train_dqn_agent
        run_config_params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "initial_balance": initial_balance,
            "window_size": window_size,
            "batch_size": batch_size,
            "episodes": episodes,
            # Add any other parameters that were part of the training setup
        }

        # Call the (soon to be adapted) train_dqn_agent function
        # This assumes train_dqn_agent will be modified to:
        # 1. Accept progress_callback and use_gpu_flag, run_config_params
        # 2. Not use input() for confirmations (these are handled by Streamlit UI before calling this)
        # 3. Check a cancellation flag (e.g., scripts.train_agent.training_cancelled)
        
        # The actual call to train_dqn_agent:
        # It needs to be adapted to take the progress_callback and use_gpu_flag.
        # For now, we are just setting up the bridge. The next step would be to
        # adapt train_dqn_agent itself.

        progress_callback({"type": "log", "level": "info", "message": f"Calling train_dqn_agent with: symbol={symbol}, timeframe={timeframe}, episodes={episodes}, use_gpu={use_gpu_flag}"})

        # This is where the actual call to the (refactored) train_dqn_agent would go.
        # For this subtask, we are just creating the bridge structure.
        # Example of how it might look *after* train_dqn_agent is refactored:
        # training_results = train_dqn_agent(
        #     symbol=symbol,
        #     timeframe=timeframe,
        #     start_date=start_date_str,
        #     end_date=end_date_str,
        #     initial_balance=initial_balance,
        #     window_size=window_size,
        #     batch_size=batch_size,
        #     episodes=episodes,
        #     output_dir=output_dir_str,
        #     use_gpu_flag=use_gpu_flag,
        #     progress_callback=progress_callback,
        #     run_config_params=run_config_params
        #     # Potentially a way to pass/check a cancellation flag if the global one is problematic
        # )

        # if training_results:
        #     progress_callback({"type": "result", "success": True, "data": training_results})
        # else:
        #     # Check if cancellation was the reason for no results
        #     # This requires train_dqn_agent or its module to communicate cancellation
        #     was_cancelled = False # Placeholder for actual cancellation check
        #     if was_cancelled:
        #         progress_callback({"type": "status", "message": "Training was cancelled by user."})
        #         progress_callback({"type": "result", "success": False, "error_message": "Training cancelled."})
        #     else:
        #         progress_callback({"type": "status", "message": "Training failed or returned no results."})
        #         progress_callback({"type": "result", "success": False, "error_message": "Training failed."})

        # For now, simulate a placeholder action
        progress_callback({"type": "log", "level": "info", "message": "Placeholder: train_dqn_agent would be called here."})
        progress_callback({"type": "status", "message": "Training simulation complete (placeholder)." })
        progress_callback({"type": "result", "success": True, "data": {"message": "Simulated training run."}})


    except Exception as e:
        logger.error(f"Error in Streamlit training bridge: {str(e)}", exc_info=True)
        progress_callback({"type": "log", "level": "error", "message": f"Bridge Error: {str(e)}"})
        progress_callback({"type": "result", "success": False, "error_message": str(e)})
        return None

if __name__ == '__main__':
    # Example of how Streamlit might use this bridge
    def dummy_streamlit_callback(update: Dict[str, Any]):
        print(f"Streamlit Update: {update}")

    print("Starting dummy training via bridge...")
    start_training_for_streamlit(
        symbol="BTCUSDT",
        timeframe="1h",
        start_date_str="2023-01-01",
        end_date_str="2023-06-01",
        initial_balance=10000,
        window_size=10,
        batch_size=64,
        episodes=5, # Small number for testing
        output_dir_str="outputs_streamlit_test",
        use_gpu_flag=False,
        progress_callback=dummy_streamlit_callback
    )
    print("Dummy training call finished.")
