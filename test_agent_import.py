import sys
import os

# Add src to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added {src_dir} to sys.path")

# Try to import the agent
try:
    from nes_copilot.agent_mvnes import MVNESAgent
    print("Successfully imported MVNESAgent")
    
    # Create an instance
    agent = MVNESAgent()
    print("Successfully created MVNESAgent instance")
    
    # Try to call the method
    result = agent.run_mvnes_trial(True, True)
    print(f"Method call result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    raise
