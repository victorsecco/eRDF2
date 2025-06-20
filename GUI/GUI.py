import tkinter as tk
from tkinter import filedialog, ttk, messagebox


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image   

from mypackages.edp_processing import ImageAnalysis, ImageProcessing, DataLoader
from mypackages.utils import bin_image, normalize_image
from mypackages.eRDF import DataProcessor

class DiffractionViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Diffraction .ser Viewer")
        self.geometry("900x700")

        # File load button
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        # Open menu
        open_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=open_menu)
        open_menu.add_command(label="Open .ser file", command=self.load_ser_file)
        open_menu.add_command(label="Open .png file", command=self.load_png_file)
        open_menu.add_command(label="Open .tif file", command=self.load_tif_file)

        # --- Analysis menu ---
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Find Center", command=self.run_find_center)
        analysis_menu.add_command(label="Apply Mask", command=self.apply_mask)
        analysis_menu.add_command(label="Azimuthal Integration", command=self.run_azimuthal_integration)
        analysis_menu.add_command(label="eRDF Analysis (F(Q), G(r))", command=self.open_erdf_window)



        # Frame navigation controls
        self.nav_frame = ttk.Frame(self)
        self.prev_button = ttk.Button(self.nav_frame, text="<< Previous", command=self.prev_frame)
        self.prev_button.grid(row=0, column=0)

        self.frame_entry = ttk.Entry(self.nav_frame, width=5)
        self.frame_entry.grid(row=0, column=1)
        self.frame_entry.insert(0, "0")

        self.go_button = ttk.Button(self.nav_frame, text="Go", command=self.go_to_frame)
        self.go_button.grid(row=0, column=2)

        self.next_button = ttk.Button(self.nav_frame, text="Next >>", command=self.next_frame)
        self.next_button.grid(row=0, column=3)

        # Matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Data storage
        self.ser_data = None
        self.num_frames = 0
        self.current_index = 0
        self.current_image = None  # In __init__()
        self.last_center = None
        self.mask_applied = False  # tracks if any mask was applied


        # Initialize the ImageAnalysis object
        self.analysis = ImageAnalysis()
        self.loader = DataLoader()
        self.image_processor = ImageProcessing(path="")


    def load_ser_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("SER files", "*.ser")])
        if not file_path:
            return

        self.ser_data, self.num_frames = self.loader.load_ser(file_path)

        if self.num_frames > 1:
            self.nav_frame.pack(pady=5)
        else:
            self.nav_frame.pack_forget()
            self.current_index = 0

        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, "0")
        self.show_frame(self.current_index)


    def load_png_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if not file_path:
            return

        img = self.loader.load_png(file_path)
        self.full_image = img
        self.current_image = bin_image(img, factor=2)
        self.nav_frame.pack_forget()
        self.display_image(self.current_image, title="PNG Image")

    def load_tif_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff")])
        if not file_path:
            return

        img = self.loader.load_tif(file_path)

        if img.dtype != np.uint16:
            print("Warning: image is not 16-bit.")

        self.full_image = img
        self.current_image = bin_image(img, factor=2)
        self.nav_frame.pack_forget()
        self.display_image(self.current_image, title="TIFF Image")

    def display_image(self, img, title="Image"):
        self.ax.clear()
        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(title)
        self.ax.axis("off")
        self.canvas.draw()

    def show_frame(self, frame_index):
        if self.ser_data is None or not (0 <= frame_index < self.num_frames):
            return
        self.current_index = frame_index
        img = self.ser_data.inav[frame_index].data
        img_binned = bin_image(img, factor=4)

        self.ax.clear()
        self.ax.imshow(img_binned, cmap="gray")
        self.ax.set_title(f"Frame {frame_index}")
        self.ax.axis("off")
        self.canvas.draw()

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(self.current_index))
            self.show_frame(self.current_index)

    def next_frame(self):
        if self.current_index < self.num_frames - 1:
            self.current_index += 1
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(self.current_index))
            self.show_frame(self.current_index)

    def go_to_frame(self):
        try:
            val = int(self.frame_entry.get())
            if 0 <= val < self.num_frames:
                self.current_index = val
                self.show_frame(val)
        except ValueError:
            pass  # Ignore invalid input

    def run_find_center(self):
        if self.full_image is None:
            print("No image loaded.")
            return

        try:
            center_x, center_y, r = self.analysis.find_center(
                self.full_image,
                r=20,
                R=200,
                threshold=80,
                edges_thresh1=255,
                edges_thresh2=10
            )
            print(f"Center found: ({center_x}, {center_y}), radius: {r}")
            self.last_center = (center_x, center_y)

            self.ax.clear()
            self.ax.imshow(self.current_image, cmap="gray")
            self.ax.plot(center_x/2, center_y/2, "ro")
            self.ax.set_title(f"Center: ({center_x}, {center_y})")
            self.ax.axis("off")
            self.canvas.draw()

        except Exception as e:
            print(f"Error running find_center: {e}")

    def run_azimuthal_integration(self):
        if self.full_image is None or self.last_center is None:
            print("Image or center not available.")
            return
        
        if not self.mask_applied:
            if messagebox.askyesno("Apply Beamstopper Mask?", "Would you like to apply beamstopper removal before integration?"):
                try:
                    img_to_integrate = self.image_processor.fixed_defects_mask(img_to_integrate, microscope="titan")
                    self.mask_applied = True
                except Exception as e:
                    print(f"Beamstopper mask failed: {e}")

            if messagebox.askyesno("Apply Custom Mask?", "Would you like to apply a general mask before integration?"):
                self.apply_mask()
                img_to_integrate = self.full_image.copy()

            

        try:
            # You can adapt binning to image size or keep a fixed value
            binning = 3000

            iq_curve, polar_image, _ = self.analysis.azimuth_integration_cv2(
                img=self.full_image,
                center=self.last_center,
                binning=binning
            )

            self.iq_curve = iq_curve  # Save for DataProcessor

            # Plot the integrated I(q)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(iq_curve, label="Azimuthal Integration")
            ax.set_title("I(q) from Azimuthal Integration")
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Intensity")
            ax.legend()
            plt.show()

        except Exception as e:
            print(f"Azimuthal integration failed: {e}")

    def apply_mask(self):
        if self.full_image is None:
            print("No image loaded.")
            return

        mask_path = filedialog.askopenfilename(title="Select Mask File", filetypes=[("Image files", "*.tif *.png *.jpg")])
        if not mask_path:
            return

        try:
            # Load and apply the mask
            mask = self.image_processor.load_mask(mask_path)
            masked_image = self.image_processor.subtract_mask(self.full_image.copy(), mask)

            # Store the real masked data for further analysis
            self.full_image = masked_image

            # Prepare a display version: replace masked pixels with max value
            display_image = masked_image.filled(masked_image.max() if hasattr(masked_image, 'filled') else 0)

            # Bin for display
            self.current_image = bin_image(display_image, factor=2)
            self.display_image(self.current_image, title="Mask Highlighted (Max Value)")

            print("Mask applied. Masked regions shown at max intensity.")
            self.mask_applied = True  # Set flag to indicate mask was applied

        except Exception as e:
            print(f"Failed to apply and highlight mask: {e}")

    def open_erdf_window(self):
        if self.iq_curve is None:
            print("You must perform azimuthal integration first.")
            return

        window = tk.Toplevel(self)
        window.title("eRDF Parameters")
        window.geometry("280x250")

        # --- Input Fields ---
        labels = [
            ("Q0 Offset", "0.0"),
            ("Q Sampling (ds)", "1e-4"),
            ("Start index", "0"),
            ("End index", str(len(self.iq_curve))),
            ("Fit region (0–1)", "0.9"),
            ("Element 1 (row,count)", "0,1"),
        ]
        entries = {}

        for i, (label, default) in enumerate(labels):
            ttk.Label(window, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=5)
            entry = ttk.Entry(window)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=10, pady=5)
            entries[label] = entry

        def run():
            try:
                q0 = float(entries["Q0 Offset"].get())
                ds = float(entries["Q Sampling (ds)"].get())
                start = int(entries["Start index"].get())
                end = int(entries["End index"].get())
                region = float(entries["Fit region (0–1)"].get())

                # Parse element entry
                row_str, count_str = entries["Element 1 (row,count)"].get().split(",")
                Elements = {1: [int(row_str), int(count_str)]}

                processor = DataProcessor(
                    data=self.iq_curve,
                    q0=q0,
                    lobato_path=None,
                    start=start,
                    end=end,
                    ds=ds,
                    Elements=Elements,
                    region=region
                )

                sq, fq = processor.SQ_PhiQ(processor.iq, damping=0.00)
                r, Gr = processor.Gr(fq, rmax=30, dr=0.02)
                processor.plot_results(fq, sq, r, Gr)


            except Exception as e:
                print(f"eRDF failed: {e}")

        ttk.Button(window, text="Run eRDF Analysis", command=run).grid(row=len(labels), column=0, columnspan=2, pady=20)





if __name__ == "__main__":
    app = DiffractionViewer()
    app.mainloop()
