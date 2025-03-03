import csv
import plotly.graph_objs as go
import plotly.io as pio

# Path to CSV file
csv_path = "/home/demir/Desktop/oct14-breathing/100um-3.csv"

# Initialize lists for data
x = []
robot_z_pos = []
linear_stage_z_pos = []

# Read CSV data
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    first_rob_pos, first_linear_stage_pos = None, None
    for row in lines:
        break
    for row in lines:
        if not first_rob_pos and not first_linear_stage_pos:
            first_rob_pos = float(row[2])
            first_linear_stage_pos = float(row[7])
        x.append(float(row[8]))
        robot_z_pos.append((float(row[2])-(first_rob_pos-0.5)))
        linear_stage_z_pos.append(float(row[7])-first_linear_stage_pos)



# Create traces for the two plots
trace1 = go.Scatter(x=x, y=robot_z_pos, mode='lines', name='Robot Z pos')
trace2 = go.Scatter(x=x, y=linear_stage_z_pos, mode='lines', name='Eye Z pos')

# Create a figure with the two traces
fig = go.Figure(data=[trace1, trace2])

# Show the plot
pio.show(fig)
