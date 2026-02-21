# hackathon_IMAC

### Setup python:
1. Configure dataset repository paths (defaults are fine, just remember to put extract the online dataset into the same place as raw_dataset)
2. Create virtual environment ```python -m venv .venv```, 
activate the environment ```/.venv/scripts/activate```, 
and install dependencies```pip install -r requirements.txt```
3. Install the dataset at https://zenodo.org/records/8219786. Specifically: anonymous_public_load_power_data_per_unit.zip
4. Run the temp_preprocessing.ipynb jupyter notebook TEMPORARY to prepare the data
5. 

### Setup demo frontend:
in frontend/usage-forecast-ui/
```npm install -D vite``` Will initialize the vite project
```npm run dev``` Will let you run the demo in localhost:5173