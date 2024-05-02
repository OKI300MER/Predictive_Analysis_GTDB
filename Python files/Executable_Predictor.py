import tkinter as tk
from tkinter import ttk

def compute_attack_metrics(nkill, nwound, target_type_weight, parameters):
    """
    Function to compute essential metrics for each incident based on specified parameters.
    """
    # Compute severity and impact
    attack_severity = min((nkill + (nwound / parameters['severity_factor'])), parameters['max_severity'])
    potential_impact = min(nkill + nwound + target_type_weight * parameters['impact_factor'], parameters['max_impact'])
    
    # Classify severity
    if attack_severity <= 5:
        severity_category = "Low"
    elif 6 <= attack_severity <= 25:
        severity_category = "Medium"
    elif 26 <= attack_severity <= 75:
        severity_category = "High"
    else:
        severity_category = "Extremely High"
    
    # Classify impact
    if potential_impact <= 10:
        impact_category = "Low"
    elif 11 <= potential_impact <= 50:
        impact_category = "Medium"
    else:
        impact_category = "High"
    
    return attack_severity, potential_impact, severity_category, impact_category

def process_data():
    nkill = float(nkill_entry.get())
    nwound = float(nwound_entry.get())
    target_type_weight = float(target_type_weight_combobox.get().split()[0])  # Extracting the numeric value from the selected string

    # Compute metrics and categories
    severity, impact, severity_category, impact_category = compute_attack_metrics(nkill, nwound, target_type_weight, parameters)
    
    # Update labels
    severity_label.config(text=f"Severity: {severity} ({severity_category})")
    impact_label.config(text=f"Impact: {impact} ({impact_category})")

# Example parameters
parameters = {
    'severity_factor': 3,  # Adjust based on your dataset
    'max_severity': 100,  # Maximum severity value (0-100%)
    'impact_factor': .05,  # Adjust based on your dataset
    'max_impact': 100,  # Maximum impact value (0-100%)
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
target_weight_options = ['5 (Business)', '7 (Police)', '3 (Private Citizens & Property)', '8 (Utilities)',
                         '8 (Military)', '1 (Violent Political Party)', '8 (Government - General)',
                         '3 (Transportation)', '1 (Tourists)', '8 (Government - Diplomatic)',
                         '5 (Religious Figures/Institutions)', '1 (Abortion Related)', '5 (Journalists & Media)',
                         '5 (NGO)', '7 (Telecommunication)', '1 (Terrorists/Non-State Militia)',
                         '6 (Educational Institution)', '8 (Airports & Aircraft)', '1 (Unknown)',
                         '7 (Maritime)', '7 (Food or Water Supply)', '1 (Other)']

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