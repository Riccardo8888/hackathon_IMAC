# hackathon_IMAC

### Setup python:
1. Configure dataset repository paths (defaults are fine, just remember to put extract the online dataset into the same place as raw_dataset)
2. Create virtual environment ```python -m venv .venv```, 
activate the environment ```/.venv/scripts/activate```, 
and install dependencies```pip install -r requirements.txt```
3. Install the dataset at https://zenodo.org/records/8219786. Specifically: anonymous_public_load_power_data_per_unit.zip
4. Run the full_preprocessing.ipynb jupyter notebook to prepare the data
5. The full pipeline is run through the full-pipeline jupyter notebook found in this directory

### Setup demo frontend:
in frontend/usage-forecast-ui/
```npm install -D vite``` Will initialize the vite project
```npm run dev``` Will let you run the demo in localhost:5173

It doesnt qualify as a demo in any sense of the word, and is mainly just a cleaner way to display plots