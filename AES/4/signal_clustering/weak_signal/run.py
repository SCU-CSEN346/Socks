import subprocess

def run_script(script_name):
    """Runs a Python script and waits for it to complete."""
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    # Run sim_matrix.py first
    run_script('sim_matrix.py')
    
    # If sim_matrix.py was successful, run pred.py
    run_script('pred.py')