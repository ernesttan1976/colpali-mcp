#!/usr/bin/env python3
"""
Patch for modifying app.py to disable flash-attention installation attempt.
Run this before starting the application if you want to disable the flash-attention install.
"""

import os
import re
import shutil

# Path to the app.py file
APP_FILE = 'app.py'

# Make a backup of the original file
backup_file = f"{APP_FILE}.backup"
if not os.path.exists(backup_file):
    shutil.copy2(APP_FILE, backup_file)
    print(f"Created backup file: {backup_file}")

# Read the app.py file
with open(APP_FILE, 'r') as f:
    content = f.read()

# Check if the install_fa2 function is being called
if '@spaces.GPU\ndef install_fa2():' in content and 'install_fa2()' in content:
    # Comment out the call to install_fa2 function
    modified_content = re.sub(
        r'(# )?install_fa2\(\)',
        '# install_fa2()  # Disabled for Docker',
        content
    )

    # Write the modified content back to app.py
    with open(APP_FILE, 'w') as f:
        f.write(modified_content)
    
    print(f"Modified {APP_FILE} to disable flash-attention installation")
else:
    print(f"No modifications needed for {APP_FILE}")