printf "%s" "BASE_DIRECTORY=" >> .env
python -c "import os; print(os.getcwd())" >> .env