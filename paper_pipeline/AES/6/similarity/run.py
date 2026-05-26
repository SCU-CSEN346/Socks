import subprocess

def run_script(script_name):
    """Runs a Python script and waits for it to complete."""
    try:
        result = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} completed successfully with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    
    run_script('dense_pred.py')

    run_script('sparse_pred.py')