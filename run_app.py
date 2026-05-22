import subprocess
import sys
import time

def run_app():
    # Start the FastAPI server
    api_process = subprocess.Popen([sys.executable, "-m", "app.Fast_api"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Starting FastAPI server...")
    
    # Wait for the API to be ready
    time.sleep(2)
    
    # Start the PySide6 GUI application
    print("Starting PySide6 GUI application...")
    gui_process = subprocess.Popen([sys.executable, "-m", "app.Widget"])

    try:
        while True:
            time.sleep(1)
            
            # If the PySide6 widget process is killed in the terminal, break
            if gui_process.poll() is not None:
                print("\nPySide6 widget process was stopped.")
                break
                
            # If the API crashes, warn user but keep the thread alive for debugging
            if api_process.poll() is not None:
                print("FastAPI backend dropped offline unexpectedly!")
                
    except KeyboardInterrupt:
        print("\nShutting down app...")
        
    finally:
        print("Cleaning up")
        # Terminate both processes if they are still running
        gui_process.terminate()
        api_process.terminate()
        
        # Wait for processes to exit
        gui_process.wait()
        api_process.wait()
        print("Terminal ready")

if __name__ == "__main__":
    run_app()

# Usage:
# python run_app.py