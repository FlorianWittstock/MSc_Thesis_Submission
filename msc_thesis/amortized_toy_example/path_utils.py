from pathlib import Path
import os

def find_git_root() -> Path:
    """
    Returns the path to the root of the git repository by walking up directories
    until a .git directory is found.
    
    Raises ValueError if no .git directory is found.
    """
    start_path = Path(os.getcwd())
    print(f'start_path: {start_path}')
    
    current_path = start_path
    while True:
        if (current_path / ".git").exists():
            return current_path
        
        # If we've reached the top-most directory without finding .git, bail out
        if current_path.parent == current_path:
            raise ValueError("No .git directory found in any parent of {}".format(start_path))
        
        # Step up one level
        current_path = current_path.parent

    
repo_base_path = find_git_root()

if __name__ == '__main__':
    print(repo_base_path)   
