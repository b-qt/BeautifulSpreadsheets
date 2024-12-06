import subprocess, sys, importlib

def import_packages(file_path: str = 'requirements.txt') -> None:
    """
        This function imports the necessary packages for the project from 
        the requirements.txt file.
    """
    try:
        with open(file_path, 'r') as req_file:
            packages = req_file.readlines() # Read the packages from the file
            for package in packages:
                package = package.strip() # Remove any leading/trailing whitespace
                if package: # Check if the package is not empty
                    importlib.import_module(package.split('==')[0]) # Import the package
                    print(f"Imported package: {package}")
    except FileNotFoundError as e:
        print(f"Failed to import packages from {file_path}\nError: {e}")
    except ImportError as e:
        print(f"Failed to import package: {e}")

# importlib.reload(sys.modules[__name__]) # Reload the module to get the new packages