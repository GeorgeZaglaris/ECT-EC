
#  Thermal Comfort Prediction & EnergyPlus Simulation Tool

This application allows you to:

- **Predict thermal comfort metrics (PMV & PPD)** using a pre-trained Machine Learning model.
- **Run EnergyPlus simulations** with `.idf` and `.epw` files to compute PMV and energy consumption.
- **Visualize** the relationship between PMV and energy consumption.
- Interact with everything via a modern **Streamlit dashboard**.

---

##  Features

- Upload `.csv` files with measured environmental data to predict PMV/PPD using an ML model.
- Upload `.idf` and `.epw` files to simulate indoor comfort and energy use via EnergyPlus.
- Graphs showing:
  - PMV vs Energy (scatter + regression)
  - PMV over time
  - Dual-axis PMV/Energy plot over time

---

##  Prerequisites

Before installation, make sure you have:

- **Python 3.10+**
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed  
  → Recommended for managing virtual environments

If you don't have Conda installed:

- Download from:
  - [Miniconda (lightweight)](https://docs.conda.io/en/latest/miniconda.html)
  - [Anaconda (full data science suite)](https://www.anaconda.com/download)
- After installation, open **Anaconda Prompt** (on Windows) or your terminal

 **EnergyPlus must be downloaded and installed manually** from [https://energyplus.net](https://energyplus.net)

---

##  Installation

Install Git LFS (Large File Support)

Before cloning the repository, install Git LFS:

Visit https://git-lfs.com

- Initialize Git LFS (once per system)

```bash
git lfs install
git clone https://github.com/GeorgeZaglaris/ECT-EC
cd ECT-EC
git lfs pull

# Create a virtual environment (recommended)
conda create -n comfort-env python=3.10
conda activate comfort-env

# Install dependencies
pip install -r requirements.txt
```

---

## Additional Requirements (for API)

If you want to run the ML model via API:

```bash
pip install fastapi uvicorn
```



---

##  Requirements

- Python 3.10+
- EnergyPlus v25.1 installed locally:
  - Example path: `C:\EnergyPlusV25-1-0`
  - Required executables:
    - `energyplus.exe`
    - `ExpandObjects.exe`
    - `PostProcess/ReadVarsESO.exe`

You may need to adjust the executable paths inside the code (`app.py`).

---

##  ML CSV File Structure

To use the ML prediction option, your CSV file must include the following columns:

| Column Name                     | Description                        |
|----------------------------------|------------------------------------|
| `Timestamp`                     | Date/time in format `%Y-%m-%d %H:%M:%S` |
| `Zone`                          | Name of the thermal zone           |
| `Air temperature (C)`           | Measured air temperature           |
| `Relative humidity (%)`         | Relative humidity in %             |
| `Air velocity (m/s)`            | Air speed                          |
| `Clo`                           | Clothing insulation value          |
| `Met`                           | Metabolic rate                     |
| `Mean Radiant Temperature(C)` or `Globe temperature (C)` | Required for MRT estimation |

---

##  IDF Output Requirements (EnergyPlus)

Your `.idf` file **must include** the following output variables/meters to generate the required analysis:

```idf
Output:Variable,
    *, Zone Thermal Comfort Fanger Model PMV, hourly;

Output:Meter,
    InteriorLights:Electricity, hourly;

Output:Meter,
    InteriorEquipment:Electricity, hourly;

Output:Meter,
    Cooling:DistrictCooling, hourly;

Output:Meter,
    Heating:DistrictHeatingWater, hourly;
```

These outputs are required to compute hourly **PMV** and total **energy consumption** in kWh.

---

##  Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

##  Optional: ML Prediction API (FastAPI)

If you want to run the ML model separately:

```bash
# Start the FastAPI server (backend must be implemented separately)
uvicorn api:app --reload
```

Then in the Streamlit app, ensure that `http://127.0.0.1:8000/predict` is accessible.

---

##  Example Visualizations

- PMV vs Energy with Linear Regression  
- PMV over time  
- PMV & Energy dual-axis line chart  

Each one is generated dynamically inside the Streamlit app after prediction or simulation.

---

##  Credits

Developed for internal testing at **Uni.systems** for evaluating indoor comfort and energy tradeoffs using ML and EnergyPlus.
