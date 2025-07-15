import os
import subprocess
import pandas as pd
import math
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
import plotly.graph_objects as go
import streamlit as st
import requests
import tempfile
import shutil


'''st.set_page_config(page_title="Thermal Comfort Tool", layout="wide")

st.title("Uni.systems")
st.header("Comfortness prediction tool for the building of Uni.systems and electric energy consumption.")
st.subheader("Options: Machine Learning or EnergyPlus Simulation")



def calculate_ppd(pmv):
    return 100 - 95 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)

def calculate_mrt(Tg, Ta, v, ε=0.95, D=0.15): #Mean Radiant Temperature
    Tg_K = Tg + 273.15
    Ta_K = Ta + 273.15
    term = Tg_K**4 + (1.1e8 * v**0.6 * (Tg_K - Ta_K)) / (ε * D**0.4)
    if term < 0:
        return None
    Tr_K = term**0.25
    return Tr_K - 273.15

def plot_common_charts(df):
    df_sorted = df.sort_values("Timestamp")

    # Regression
    X = df["Energy_kWh"]
    y = df["PMV"]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    df["PMV_pred"] = model.predict(X_const)

    st.subheader("Diagrams")

    # 1. Scatter plot with regression
    fig1 = px.scatter(df, x="Energy_kWh", y="PMV", title="Energy vs PMV with Regression Line",
                      labels={"Energy_kWh": "Energy (kWh)", "PMV": "PMV"}, hover_data=["Timestamp"])
    fig1.add_scatter(x=df["Energy_kWh"], y=df["PMV_pred"], mode="lines", name="Regression", line=dict(color="red"))
    st.plotly_chart(fig1)

    # 2. PMV over time
    fig2 = px.line(df_sorted, x="Timestamp", y="PMV", title="PMV vs Time", hover_data=["Energy_kWh"])
    fig2.update_traces(mode="lines+markers")
    st.plotly_chart(fig2)

    # 3. Dual axis plot
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_sorted["Timestamp"], y=df_sorted["PMV"],
                              name="PMV", yaxis='y1', mode='lines+markers', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=df_sorted["Timestamp"], y=df_sorted["Energy_kWh"],
                              name="Energy (kWh)", yaxis='y2', mode='lines+markers', line=dict(color='green')))

    fig3.update_layout(
        title="PMV & Power Consumption in relation to time",
        xaxis=dict(title="Time"),
        yaxis=dict(title="PMV", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="Energy (kWh)", titlefont=dict(color="green"),
                    tickfont=dict(color="green"), overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig3)

def process_ml_dataframe(df):
    #check if all the required variables are presenting
    df = df.loc[:, ~df.columns.duplicated()]
    required = ['Zone', 'Air temperature (C)', 'Relative humidity (%)', 'Air velocity (m/s)', 'Clo', 'Met']
    has_mrt = 'Mean Radiant Temperature(C)' in df.columns
    has_globe = 'Globe temperature (C)' in df.columns

    if not all(col in df.columns for col in required) or not (has_mrt or has_globe):
        st.error("Missing columns. Required: " + ", ".join(required) +
                 ", and either 'Mean Radiant Temperature(C)' or 'Globe temperature (C)'")
        return

    st.subheader("Uploaded Data")
    st.dataframe(df)

    if st.button("Predict PMV & PPD"):
        results = []
        progress_bar = st.progress(0)

        for i, row in df.iterrows():
            if 'Mean Radiant Temperature(C)' in df.columns and pd.notna(row.get('Mean Radiant Temperature(C)')):
                mrt = row['Mean Radiant Temperature(C)']
            elif all(col in df.columns for col in ['Globe temperature (C)', 'Air temperature (C)', 'Air velocity (m/s)']):
                mrt = calculate_mrt(row['Globe temperature (C)'], row['Air temperature (C)'], row['Air velocity (m/s)'])
            else:
                mrt = None

            if mrt is None:
                results.append({"PMV": None, "PPD": None})
                progress_bar.progress((i + 1) / len(df))
                continue

            payload = {
                "air_temperature": row["Air temperature (C)"],
                "mean_radiant_temperature": mrt,
                "relative_humidity": row["Relative humidity (%)"],
                "air_velocity": row["Air velocity (m/s)"],
                "clo": row["Clo"],
                "met": row["Met"]
            }

            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                result = response.json()
                pmv = result["predicted_PMV"]
                ppd = calculate_ppd(pmv)
                results.append({"PMV": pmv, "PPD": ppd})
            except:
                results.append({"PMV": None, "PPD": None})

            progress_bar.progress((i + 1) / len(df))

        result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        st.success("Prediction completed.")
        st.dataframe(result_df)

        if "Timestamp" in result_df.columns:
            result_df["Timestamp"] = pd.to_datetime(result_df["Timestamp"], errors="coerce")
            result_df["Hour"] = result_df["Timestamp"].dt.hour
            result_df["Date"] = result_df["Timestamp"].dt.date

        valid_results = result_df.dropna(subset=["PMV", "PPD"])

        if "Hour" in valid_results.columns:
            st.subheader("Average PMV & PPD per Hour")
            hourly_df = valid_results.groupby(["Zone", "Hour"])[["PMV", "PPD"]].mean().reset_index()
            st.dataframe(hourly_df)
            fig = px.line(hourly_df, x="Hour", y="PMV", color="Zone", markers=True,
                          title="Hourly PMV per Zone")
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(24))))
            st.plotly_chart(fig)

        st.subheader("Average PMV & PPD per Zone")
        zone_df = valid_results.groupby("Zone")[["PMV", "PPD"]].mean().reset_index()
        st.dataframe(zone_df)
        st.plotly_chart(px.bar(zone_df, x="Zone", y=["PMV", "PPD"], barmode="group",
                               title="Thermal Comfort per Zone"))

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download ML Predictions", csv, file_name="ml_predictions.csv")

def run_energyplus_simulation(idf_file, epw_file, output_dir):
    
    #change the path if necessary based on the installation 

    energyplus_exe = r"C:\EnergyPlusV25-1-0\energyplus.exe"
    readvars_exe = r"C:\EnergyPlusV25-1-0\PostProcess\ReadVarsESO.exe"
    expand_objects_exe = r"C:\EnergyPlusV25-1-0\ExpandObjects.exe"

    with tempfile.TemporaryDirectory() as tmpdir:
        idf_path = os.path.join(tmpdir, "in.idf")
        epw_path = os.path.join(tmpdir, "weather.epw")

        with open(idf_path, "wb") as f: f.write(idf_file.read())
        with open(epw_path, "wb") as f: f.write(epw_file.read())

        # Run ExpandObjects(req for HVAC variables)
        subprocess.run([expand_objects_exe], cwd=tmpdir, check=True)

        expanded_path = os.path.join(tmpdir, "expanded.idf")
        if not os.path.exists(expanded_path):
            raise FileNotFoundError("Expanded IDF file not created. Check ExpandObjects output.")

        sim_output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(sim_output_dir, exist_ok=True)

        subprocess.run([
            energyplus_exe,
            "-w", epw_path,
            "-d", sim_output_dir,
            "-r", expanded_path
        ], check=True)

        # Convert ESO to CSV
        rvi_path = os.path.join(sim_output_dir, "output.rvi")
        with open(rvi_path, "w") as f:
            f.write("eplusout.eso\n")
            f.write("eplusout.csv\n")
            f.write("0\n")
            f.write("Zone Thermal Comfort Fanger Model PMV\n")
            f.write("InteriorLights:Electricity\n")
            f.write("InteriorEquipment:Electricity\n")
            f.write("Cooling:DistrictCooling\n")
            f.write("Heating:DistrictHeatingWater\n")
            f.write("0\n")

        subprocess.run([readvars_exe, "output.rvi"], cwd=sim_output_dir, check=True)

        csv_path = os.path.join(sim_output_dir, "eplusout.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError("CSV file not generated.")

        final_csv_path = os.path.join(output_dir, "eplusout.csv")
        shutil.copy(csv_path, final_csv_path)
        return final_csv_path

# choise between ml or e+ sim

mode = st.radio("Choose a Method:", ["Machine Learning", "EnergyPlus Simulation"])

if mode == "Machine Learning":
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        process_ml_dataframe(df)

elif mode == "EnergyPlus Simulation":
    idf_file = st.file_uploader("Upload IDF File", type="idf")
    epw_file = st.file_uploader("Upload EPW Weather File", type="epw")
    output_dir = st.text_input("Choose Output Directory(Paste your path)", placeholder=os.path.join(os.getcwd(), "eplus_outputs"))

    if st.button("Run EnergyPlus Simulation"):
        if not idf_file or not epw_file:
            st.error("Please upload both an IDF and EPW file.")
        elif not output_dir.strip():
          st.error("Please enter a valid output directory path.")
        else:
            os.makedirs(output_dir, exist_ok=True)
            try:
                csv_path = run_energyplus_simulation(idf_file, epw_file, output_dir)
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                df["Date/Time"] = df["Date/Time"].str.strip()
                df["Timestamp"] = pd.to_datetime("2025/" + df["Date/Time"], format="%Y/%m/%d %H:%M:%S", errors='coerce') #change the year if necessary

                df.rename(columns={
                    df.columns[1]: "PMV",
                    df.columns[2]: "Lights_J",
                    df.columns[3]: "Equipment_J",
                    df.columns[4]: "Cooling_J",
                    df.columns[5]: "Heating_J"
                }, inplace=True)

                #Sum of Energy in kwh
                df["Energy_kWh"] = (df["Lights_J"] + df["Equipment_J"] + df["Cooling_J"] + df["Heating_J"]) / 3.6e6
                valid_df = df.dropna(subset=["PMV", "Energy_kWh"])

                st.success("Simulation completed and results processed.")
                st.dataframe(valid_df)
                plot_common_charts(valid_df)

                st.download_button("Download Results CSV", valid_df.to_csv(index=False).encode("utf-8"),
                                   file_name="eplus_pmv_energy.csv")
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                '''


# Load the trained PyCaret model
model = load_model('my_model_et')

st.set_page_config(page_title="Thermal Comfort Tool", layout="wide")

st.title("Uni.systems")
st.header("Comfortness prediction tool for the building of Uni.systems and electric energy consumption.")
st.subheader("Options: Machine Learning")

def calculate_ppd(pmv):
    return 100 - 95 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)

def calculate_mrt(Tg, Ta, v, ε=0.95, D=0.15):  # Mean Radiant Temperature
    Tg_K = Tg + 273.15
    Ta_K = Ta + 273.15
    term = Tg_K**4 + (1.1e8 * v**0.6 * (Tg_K - Ta_K)) / (ε * D**0.4)
    if term < 0:
        return None
    Tr_K = term**0.25
    return Tr_K - 273.15

def plot_common_charts(df):
    df_sorted = df.sort_values("Timestamp")
    X = df["Energy_kWh"]
    y = df["PMV"]
    X_const = sm.add_constant(X)
    model_ols = sm.OLS(y, X_const).fit()
    df["PMV_pred"] = model_ols.predict(X_const)

    st.subheader("Diagrams")

    fig1 = px.scatter(df, x="Energy_kWh", y="PMV", title="Energy vs PMV with Regression Line",
                      labels={"Energy_kWh": "Energy (kWh)", "PMV": "PMV"}, hover_data=["Timestamp"])
    fig1.add_scatter(x=df["Energy_kWh"], y=df["PMV_pred"], mode="lines", name="Regression", line=dict(color="red"))
    st.plotly_chart(fig1)

    fig2 = px.line(df_sorted, x="Timestamp", y="PMV", title="PMV vs Time", hover_data=["Energy_kWh"])
    fig2.update_traces(mode="lines+markers")
    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_sorted["Timestamp"], y=df_sorted["PMV"],
                              name="PMV", yaxis='y1', mode='lines+markers', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=df_sorted["Timestamp"], y=df_sorted["Energy_kWh"],
                              name="Energy (kWh)", yaxis='y2', mode='lines+markers', line=dict(color='green')))

    fig3.update_layout(
        title="PMV & Power Consumption in relation to time",
        xaxis=dict(title="Time"),
        yaxis=dict(title="PMV", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="Energy (kWh)", titlefont=dict(color="green"),
                    tickfont=dict(color="green"), overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig3)

def process_ml_dataframe(df):
    df = df.loc[:, ~df.columns.duplicated()]
    required = ['Zone', 'Air temperature (C)', 'Relative humidity (%)', 'Air velocity (m/s)', 'Clo', 'Met']
    has_mrt = 'Mean Radiant Temperature(C)' in df.columns
    has_globe = 'Globe temperature (C)' in df.columns

    if not all(col in df.columns for col in required) or not (has_mrt or has_globe):
        st.error("Missing columns. Required: " + ", ".join(required) +
                 ", and either 'Mean Radiant Temperature(C)' or 'Globe temperature (C)'")
        return

    st.subheader("Uploaded Data")
    st.dataframe(df)

    if st.button("Predict PMV & PPD"):
        results = []
        progress_bar = st.progress(0)

        for i, row in df.iterrows():
            if has_mrt and pd.notna(row.get('Mean Radiant Temperature(C)')):
                mrt = row['Mean Radiant Temperature(C)']
            elif has_globe:
                mrt = calculate_mrt(row['Globe temperature (C)'], row['Air temperature (C)'], row['Air velocity (m/s)'])
            else:
                mrt = None

            if mrt is None:
                results.append({"PMV": None, "PPD": None})
                progress_bar.progress((i + 1) / len(df))
                continue

            input_data = pd.DataFrame([{
                "Air temperature (C)": row["Air temperature (C)"],
                "Mean Radiant Temperature(C)": mrt,
                "Relative humidity (%)": row["Relative humidity (%)"],
                "Air velocity (m/s)": row["Air velocity (m/s)"],
                "Clo": row["Clo"],
                "Met": row["Met"]
            }])

            try:
                output = predict_model(model, data=input_data)
                pmv = output['prediction_label'][0]
                ppd = calculate_ppd(pmv)
                results.append({"PMV": pmv, "PPD": ppd})
            except Exception:
                results.append({"PMV": None, "PPD": None})

            progress_bar.progress((i + 1) / len(df))

        result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
        st.success("Prediction completed.")
        st.dataframe(result_df)

        if "Timestamp" in result_df.columns:
            result_df["Timestamp"] = pd.to_datetime(result_df["Timestamp"], errors="coerce")
            result_df["Hour"] = result_df["Timestamp"].dt.hour
            result_df["Date"] = result_df["Timestamp"].dt.date

        valid_results = result_df.dropna(subset=["PMV", "PPD"])

        if "Hour" in valid_results.columns:
            st.subheader("Average PMV & PPD per Hour")
            hourly_df = valid_results.groupby(["Zone", "Hour"])[["PMV", "PPD"]].mean().reset_index()
            st.dataframe(hourly_df)
            fig = px.line(hourly_df, x="Hour", y="PMV", color="Zone", markers=True,
                          title="Hourly PMV per Zone")
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(24))))
            st.plotly_chart(fig)

        st.subheader("Average PMV & PPD per Zone")
        zone_df = valid_results.groupby("Zone")[["PMV", "PPD"]].mean().reset_index()
        st.dataframe(zone_df)
        st.plotly_chart(px.bar(zone_df, x="Zone", y=["PMV", "PPD"], barmode="group",
                               title="Thermal Comfort per Zone"))

        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download ML Predictions", csv, file_name="ml_predictions.csv")

# Streamlit file uploader for CSV input
uploaded_file = st.file_uploader("Upload your CSV data for prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    process_ml_dataframe(df)

