import os
import shutil
import subprocess
import stat

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    """
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def reset_git():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    git_dir = os.path.join(project_dir, ".git")

    print(f"[*] Resetting Git repository in: {project_dir}")

    # 1. Remove existing .git directory
    if os.path.exists(git_dir):
        print("    Found existing .git directory. Removing...")
        try:
            shutil.rmtree(git_dir, onerror=on_rm_error)
            print("    [+] Removed .git directory.")
        except Exception as e:
            print(f"    [-] Error removing .git directory: {e}")
            print("    Please manually delete the hidden .git folder and run this script again.")
            return
    else:
        print("    [!] No .git directory found. Skipping removal.")

    # 2. Initialize new git repository
    print("    Initializing new git repository...")
    try:
        subprocess.check_call(["git", "init"], cwd=project_dir)
        print("    [+] Git initialized.")
    except FileNotFoundError:
        print("    [-] 'git' command not found. Please ensure Git is installed and in your PATH.")
        return
    except subprocess.CalledProcessError as e:
        print(f"    [-] Error initializing git: {e}")
        return

    # 3. Add all files
    print("    Adding files to staging...")
    try:
        subprocess.check_call(["git", "add", "."], cwd=project_dir)
        print("    [+] Files added.")
    except subprocess.CalledProcessError as e:
        print(f"    [-] Error adding files: {e}")
        return

    # 4. Initial commit
    print("    Creating initial commit...")
    try:
        subprocess.check_call(["git", "commit", "-m", "Initial commit for FlowTrace"], cwd=project_dir)
        print("    [+] Initial commit created.")
    except subprocess.CalledProcessError as e:
        print(f"    [-] Error committing: {e}")
        return

    print("\n--- Success! ---")
    print("To push to your new GitHub repository, run the following commands in your terminal:")
    print("1. Create a new repository on GitHub named 'FlowTrace'")
    print("2. Run: git branch -M main")
    print("3. Run: git remote add origin https://github.com/<YOUR_USERNAME>/FlowTrace.git")
    print("4. Run: git push -u origin main")

if __name__ == "__main__":
    reset_git()