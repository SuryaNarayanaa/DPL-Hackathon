import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot(csv_folder):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    if not csv_files:
        print("No CSV files found in folder:", csv_folder)
        return

    # Process each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Use the first two columns for plotting
            if df.shape[1] < 2:
                print(f"Skipping {csv_file}: Less than two columns found.")
                continue

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # Create scatter plot
            plt.figure()
            plt.scatter(x, y)
            plt.title(f"Scatter plot of {os.path.basename(csv_file)}")
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            
            # Save plot image next to CSV file
            plot_filename = os.path.splitext(csv_file)[0] + '.png'
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    # Set the folder where CSV files are located.
    csv_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DPL_Datasets'))
    analyze_and_plot(csv_directory)
