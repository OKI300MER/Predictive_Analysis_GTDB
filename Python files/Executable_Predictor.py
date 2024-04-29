import tkinter as tk
from tkinter import ttk
import pandas as pd

def compute_attack_metrics(nkill, nwound, target_type_weight, parameters):
    """
    Function to compute essential metrics for each incident based on specified parameters.
    """
    attack_severity = nkill + nwound / parameters['severity_factor']
    potential_impact = nkill + target_type_weight * parameters['impact_factor']
    return attack_severity, potential_impact

def process_data():
    nkill = float(nkill_entry.get())
    nwound = float(nwound_entry.get())
    target_type_weight = float(target_type_weight_combobox.get().split()[0])  # Extracting the numeric value from the selected string

    severity, impact = compute_attack_metrics(nkill, nwound, target_type_weight, parameters)
    
    severity_label.config(text=f"Severity: {severity}")
    impact_label.config(text=f"Impact: {impact}")

# Example parameters
parameters = {
    'severity_factor': 3.0,
    'impact_factor': 0.5
}

# Create GUI
root = tk.Tk()
root.title("Attack Metrics Calculator")

main_frame = ttk.Frame(root, padding="20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

nkill_label = ttk.Label(main_frame, text="Number of Fatalities:")
nkill_label.grid(column=0, row=0, sticky=tk.W)

nkill_entry = ttk.Entry(main_frame)
nkill_entry.grid(column=1, row=0)

nwound_label = ttk.Label(main_frame, text="Number of Injuries:")
nwound_label.grid(column=0, row=1, sticky=tk.W)

nwound_entry = ttk.Entry(main_frame)
nwound_entry.grid(column=1, row=1)

target_type_weight_label = ttk.Label(main_frame, text="Target Type Weight:")
target_type_weight_label.grid(column=0, row=2, sticky=tk.W)

# Options for target type weight
target_weight_options = ['0.5 (Business)', '0.7 (Police)', '0.3 (Private Citizens & Property)', '0.8 (Utilities)',
                         '0.8 (Military)', '0.1 (Violent Political Party)', '0.8 (Government - General)',
                         '0.3 (Transportation)', '0.1 (Tourists)', '0.8 (Government - Diplomatic)',
                         '0.3 (Religious Figures/Institutions)', '0.1 (Abortion Related)', '0.5 (Journalists & Media)',
                         '0.5 (NGO)', '0.7 (Telecommunication)', '0.1 (Terrorists/Non-State Militia)',
                         '0.6 (Educational Institution)', '0.87 (Airports & Aircraft)', '0.1 (Unknown)',
                         '0.7 (Maritime)', '0.7 (Food or Water Supply)', '0.1 (Other)']

target_type_weight_combobox = ttk.Combobox(main_frame, values=target_weight_options)
target_type_weight_combobox.grid(column=1, row=2)
target_type_weight_combobox.current(0)  # Set default value

process_button = ttk.Button(main_frame, text="Process Data", command=process_data)
process_button.grid(column=0, row=3, columnspan=2)

severity_label = ttk.Label(main_frame, text="Severity: ")
severity_label.grid(column=0, row=4, columnspan=2, sticky=tk.W)

impact_label = ttk.Label(main_frame, text="Impact: ")
impact_label.grid(column=0, row=5, columnspan=2, sticky=tk.W)

root.mainloop()