#@title 1.1. Classes e funções
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize

class DataProcessor:
    def __init__(self, data, q0, lobato_path, start, end, ds, Elements, region):
        # Assuming 'data_df' is a DataFrame with multiple columns of data
        self.data = data
        self.start = start
        self.end = end
        self.ds = ds
        self.lobato = "src\packages\Lobato_2014.txt" if lobato_path is None else lobato_path
        self.region = region
        self.q0 = q0

        total = []
        for element in Elements:
            if Elements[element]:
                total.append(Elements[element][1])
        soma = sum(total)

        for element in Elements:
            if Elements[element]:
                Elements[element].append(Elements[element][1] / soma)
        self.Elements = Elements

        # Load and process data in the constructor
        self.x, self.iq, self.q, self.s, self.s2 = self.load_and_process_data(data, q0)
        self.lobato_factors = self.calculate_Lobato_Factors()
        self.fq_sq, self.gq, self.fqfit, self.iqfit = self.calculate_fq_gq()
        self.N, self.C, self.autofit = self.calculate_N_and_parameters(region=self.region)

    def load_and_process_data(self, data_column, q0):
        # Modify this method to process a single column of data
        # 'data_column' is a pandas Series representing one column of your data
        Iq = np.array(data_column)
        x = np.arange(0, len(Iq), 1)
        x, iq = x[self.start:self.end], Iq[self.start:self.end]
        q = x * self.ds * 2 * math.pi + q0
        s = q / (2 * math.pi)
        s2 = s ** 2
        return x, iq, q, s, s2

    def calculate_Lobato_Factors(self):
        FACTORS = []
        Lobato_Factors = np.empty(shape=(0))
        df = pd.read_csv(self.lobato, header=None)
        counter = 0
        for element in self.Elements:
            if self.Elements[element]:
                FACTORS.append(np.array(df.iloc[self.Elements[element][0]]))

        for LF in FACTORS:
            Lobato_Factors = np.append(Lobato_Factors,
                                       (((LF[0] * (self.s2 * LF[5] + 2) / (self.s2 * LF[5] + 1) ** 2)) +
                                        ((LF[1] * (self.s2 * LF[6] + 2) / (self.s2 * LF[6] + 1) ** 2)) +
                                        ((LF[2] * (self.s2 * LF[7] + 2) / (self.s2 * LF[7] + 1) ** 2)) +
                                        ((LF[3] * (self.s2 * LF[8] + 2) / (self.s2 * LF[8] + 1) ** 2)) +
                                        ((LF[4] * (self.s2 * LF[9] + 2) / (self.s2 * LF[9] + 1) ** 2))))
        Lobato_Factors = Lobato_Factors.reshape(len(FACTORS), len(self.x))

        return Lobato_Factors

    def calculate_fq_gq(self):
        fq = np.empty(shape=(0))
        gq = np.empty(shape=(0))


        for i in range(0, len(self.lobato_factors)):
            if self.Elements[i + 1]:
                fq = np.append(fq, self.lobato_factors[i] * self.Elements[i + 1][2])
                gq = np.append(gq, (self.lobato_factors[i] ** 2) * self.Elements[i + 1][2])

        fq_sq = np.sum(fq.reshape(len(self.Elements), len(self.x)), axis=0)
        fq_sq = fq_sq ** 2
        gq = np.sum(gq.reshape(len(self.Elements), len(self.x)), axis=0)
        fqfit = gq[self.end - (self.start+1)]
        iqfit = self.iq[self.end - (self.start+1)]

        return fq_sq, gq, fqfit, iqfit

    def calculate_N_and_parameters(self, region=0):
        interval = int(region*len(self.x))
        wi = np.ones_like(self.x[interval:])

        a1 = np.sum(self.gq[interval:] * self.iq[interval:])
        a2 = np.sum(self.iq[interval:] * self.fqfit)
        a3 = np.sum(self.gq[interval:] * self.iqfit)
        a4 = np.sum(wi[interval:]) * self.fqfit * self.iqfit
        a5 = np.sum(self.gq[interval:] ** 2)
        a6 = 2 * np.sum(self.gq[interval:]) * self.fqfit
        a7 = np.sum(wi[interval:]) * self.fqfit * self.fqfit

        N = (a1 - a2 - a3 + a4) / (a5 - a6 + a7)

        # Fitting Parameters
        C = self.iqfit - N * self.fqfit
        autofit = N * self.gq + C

        return N, C, autofit

    def calculate_SQ_PhiQ(self, iq, damping):
        sq = (((iq - self.autofit)) / (self.N * self.fq_sq)) + 1
        fq = (((iq - self.autofit) * self.s) / (self.N * self.fq_sq)) * np.exp(-self.s2 * damping)

        return sq, fq


    def calculate_Gr_Lorch(self, fq, rmax, dr, a, b):
        Gr = np.zeros_like(self.q)
        #r = np.linspace(dr, rmax, self.end-self.start)
        r = np.arange(dr, dr*(self.end-self.start)+dr, dr)
        for i, r_step in enumerate(r):
            delta = (math.pi/self.q.max()) * (1-np.exp(-abs(r_step-a)/b))
            lorch = np.sin(self.q * delta)/(self.q * delta)
            integrand = 8 * lorch * math.pi * fq * np.sin(self.q * r_step)
            Gr[i] = np.trapz(integrand, self.s)

        return r, Gr

    def calculate_Gr_Lorch_arctan(self, fq, rmax, dr, a, b, c):
        Gr = np.zeros_like(self.q)
        r = np.linspace(dr, rmax, self.end-self.start)
        for i, r_step in enumerate(r):
            delta = (math.pi / self.q.max()) * (
                    (1 - np.exp(-abs(r_step - a) / b)) +
                    (1 / 2 + 1 / math.pi * np.arctan(r_step - c / (c / (2 * math.pi))) * r_step ** (1 / 2))
                    )
            lorch = np.sin(self.q * delta)/(self.q * delta)
            integrand = 8 * lorch * math.pi * fq * np.sin(self.q * r_step)
            Gr[i] = np.trapz(integrand, self.s)

        return r, Gr

    def low_r_correction(self, Gr, nd, r, r_cut, scale_factor = 1):

        """
        Correction of the low frequency signal in the fq that generates 
        
        Parameters:
        - Gr: pair distribution function calculated after corrections
        - r_values: array of evenly spaced distance values
        - fq_direct: fq calculated from total scattering
        - density: number density in number of atoms per cubic angstrom  
        
        Returns:
        - fq_inverse: recalculated fq (the scale is not yet understood and is not optimized, a z-score
        normalization is needed)
        """

        number_density_line = -4 * math.pi * nd * r * scale_factor
        Gr_low_r = np.where(r < r_cut, Gr, 0)
        Gr = np.where(r < r_cut, number_density_line, Gr)
        return Gr, Gr_low_r


    def cut_Gr_spherical(self, Gr, r_values, diameter):
        Gr = Gr * (1 - ((3 / 2) * (r_values / diameter)) + (0.5 * (r_values / diameter) ** 3)) * np.exp(-((r_values * 0.2 ** 2) / 2))
        return Gr

    def inverse_fourier_transform(self, Gr, r_values):

        """
        Calculation of the inverse Fourier transform using the numpy numerical trapezoidal integration method
        
        Parameters:
        - Gr: pair distribution function calculated after corrections
        - r_values: array of evenly spaced distance values
        - fq_direct: fq calculated from total scattering
        - density: number density in number of atoms per cubic angstrom  
        
        Returns:
        - fq_inverse: recalculated fq (the scale is not yet understood and is not optimized, a z-score
        normalization is needed)
        """

        # Initialize S(Q) to zeros
        fq_inverse = np.zeros_like(self.q)

        # Perform the inverse Fourier transform
        for i, dq in enumerate(self.q):
            integrand = Gr * np.sin(dq * r_values)
            fq_inverse[i] = np.trapz(integrand, r_values)
        #fq_inverse = fq_inverse*normalization_factor
        #if sum(fq_direct) != 0:
        #    fq_inverse = sum(fq_inverse) / sum(fq_direct) * fq_inverse
        #    return fq_inverse
        #else:
        return fq_inverse

    def calculate_IQ(self, fq, damping):
        """
        Reverse calculation of the total scattering intensity based on the calculated fq
        
        Parameters:
        - fq: inverse calculated fq after gr corrections
        - damping: damping parameter used to make high-q signal less expressive
        
        Returns:
        - iq: recalculated total scattering intensity
        """
        iq = fq * self.N * self.fq_sq
        iq = iq / (self.s*np.exp(-self.s2*damping))
        iq = iq + self.autofit
        return iq

    def plot_results(self, fq, fq2, Gr0, r, Gr1, rw):
        f, ax = plt.subplots(1, 3, figsize=(14, 5))

        # Plotting I(Q) and Fit
        line1, = ax[0].plot(self.q, self.autofit)
        line2, = ax[0].plot(self.q, self.iq)
        ax[0].legend([line1, line2], ["Fit", "I(Q)"])
        #ax[0].text(5, 600000, 'N: ' f'{int(self.N)}')
        ax[0].set_xlabel("Q ($\AA^{-1}$)")
        ax[0].set_ylabel("Intensity")
        ax[0].title.set_text('Fitting I(Q)')

        # Plotting S(Q)
        line3, = ax[1].plot(self.q, fq, label = "$\phi(Q)$" )
        #line4, = ax[1].plot(self.q, fq2, label = "S(Q) filtered")
        ax[1].set_xlabel("Q ($\AA^{-1}$)")
        ax[1].set_ylabel("$\phi(Q)$")
        #ax[1].set_xlim(11.5,13.5)
        #ax[1].set_ylim(0.4,0.6)
        ax[1].legend()
        ax[1].title.set_text('Calculating $\phi(Q)$')

        # Plotting G(r)
        line5, = ax[2].plot(r, Gr0, label = "G(r)")
        line6, = ax[2].plot(r, Gr1, label = "G(r) Lorch")
        #ax[2].text(5, 0.8*Gr0.max(), 'Rw = 'f'{rw:.2f}')
        ax[2].set_xlabel("r ($\AA$)")
        ax[2].set_ylabel("G(r)")
        ax[2].set_xlim([0, 10])
        ax[2].title.set_text('Calculating G(r)')
        ax[2].legend()
        plt.subplots_adjust(hspace=1)
        f.tight_layout()


        plt.show()


    def save_to_csv(self, data, file_path, separator, x_name, y_name, out='pdfgui'):
        """
        Saves selected data to .csv format

        Parameters:
        - data: tuple of data comprising x,y 
        - file_path: folder in which the file will be saved
        - name: the base name of the output file
        - x, y: names of the columns in the .csv file 
        - separator: delimiter for the CSV output
        - out: formatting the file for input in the discus of pdfgui softwares
        """
        data = pd.DataFrame(np.transpose(np.array(data)))
        if out == 'discus':
            data.rename(columns={0: x_name, 1: y_name}, inplace=True)
            data[f'd{x}'] = data[x] * 0
            data[f'd{y}'] = abs(data[y] / 20)
            data.to_csv(f'{file_path}.csv', sep=separator, float_format="%.10f", index=False)
        else:
            data.to_csv(f'{file_path}.csv', sep=separator, index=False, header=False)

def calculate_Gr(q, fq, rmax, dr):
        Gr = []
        r = np.arange(0, rmax, dr)

        for i, r_step in enumerate(r):
            integrand = 8 * math.pi * fq * np.sin(q * r_step)
            Gr.append(np.trapz(integrand, q/(2* np.pi)))
        # Convert lists to numpy arrays for consistency
        r = np.array(r, dtype=np.float64)
        Gr = np.array(Gr, dtype=np.float64)

        return r, Gr/(2 * np.pi)

def q_to_two_theta(q_calibration, pixel_data, wavelength_nm):
    """
    Converts pixel data from Q space to two-theta angles using a given Q space calibration factor.
    
    Parameters:
    - q_calibration: The Q space calibration factor.
    - pixel_data: An array of pixel indices or measurements to be converted.
    - wavelength_nm: The wavelength of the X-rays in nanometers.
    
    Returns:
    - two_theta: An array of two-theta angles in degrees corresponding to the provided pixel data.
    """
    # Convert the wavelength to the same unit as q_calibration, if needed (nm to the unit of q_calibration)
    wavelength = wavelength_nm  # Assuming q_calibration is based on nm units
    
    # Calculate 1/d from pixel data using the Q space calibration factor
    one_over_d = pixel_data * (q_calibration / (2 * np.pi))
    
    # Calculate two-theta angles from 1/d values
    two_theta = 2 * np.degrees(np.arcsin(wavelength * one_over_d / 2))
    
    return two_theta


def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # fs = sample rate, Hz
    normal_cutoff = cutoff / nyq # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz / Nyquist Frequency

    # Get the filter coefficients
    b, a = butter(order, # sin wave can be approx represented as quadratic
                  normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def read_discus_fit_file(path):
    # Read file ignoring comment lines (those starting with '#')
    with open(path, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#') and line.strip() != '']
    
    # Split lines by whitespace and remove empty tokens
    data = [remove_empty_strings(line.strip().split()) for line in lines]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Assign columns depending on how many there are
    if df.shape[1] == 4:
        df.columns = ['r', 'gr', 'dr', 'dgr']
    elif df.shape[1] == 2:
        df.columns = ['r', 'gr']
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")
    
    return df.astype(float)

def calculate_rw(obs, calc, scaling = 1):
    #Calculate residuals metric from experimental and calculated data
    obs = obs * scaling
    return math.sqrt(sum((obs-calc)**2)/sum(obs**2))

def remove_empty_strings(lst):
    return [element for element in lst if element != ""]

# The wrapper function for minimize
def optimize_constant(grob, calc, initial_guess=1):
    # Objective function to minimize, takes only constant as argument
    objective = lambda constant: calculate_rw(grob, calc, constant)
    # Run the optimization
    result = minimize(objective, initial_guess)
    return result.x  # This returns the optimized constant