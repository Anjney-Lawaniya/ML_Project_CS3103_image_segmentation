#!/bin/bash

#!/bin/bash

echo "Installing dependencies..."

if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "Virtual environment not found!"
    exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo "Dependencies installed successfully!"