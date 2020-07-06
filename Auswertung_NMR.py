import numpy as np  #Modul für wissenschafftliche rechnungen
import matplotlib.pyplot as plt # Visualiesierung der Ergebnisse
from scipy.optimize import curve_fit, minimize

data = np.loadtxt("gb1_fN.txt", skiprows= 1) #einlesen der txt Datei

def fit_func(temp_cel, H, T_m):
    y_N_fit = np.polyfit(data[0:7, 0], data[0:7, 1], 1)
    y_U_fit = np.polyfit(data[-3:, 0], data[-3:, 1], 1)
    y_N = y_N_fit[0]*temp_cel+ y_N_fit[1]
    y_U = y_U_fit[0] * temp_cel + y_U_fit[1]
    temp_kel = temp_cel + 273.15
    c = 4.3  # *1000
    R = 8.314462
    A_first = (-((H / R) * ((1/temp_kel) - (1/T_m))))
    A_sec = (1 - (T_m/ temp_kel) - np.log(T_m/temp_kel))
    A =A_first  - ((c/ R) * A_sec)

    counter = (y_N + (y_U * np.exp(A)))
    denominator = (1+ np.exp(A))
    #print(counter, denominator, counter / denominator)
    return counter / denominator

y_N_fit = np.polyfit(data[0:7, 0], data[0:7, 1], 1)
y_U_fit = np.polyfit(data[-3:, 0], data[-3:, 1], 1)

bounds = ([-np.inf, 300],[np.inf, 400])
popt, pcov = curve_fit(fit_func, data[:,0],data[:,1], bounds = bounds )
print(f"popt: {popt} \n"
         f"pcov: {pcov}" )
result = []

for i in range(len(data[:,0])):
    res = fit_func(data[i,0], popt[0],popt[1])
    result.append(res)



plt.plot(data[:,0], result, "-o", label= "Polynomial fit ")
plt.plot(data[:,0], data[:,1], "-o", label=  "raw data") # Plot von Temperatur zum Meswert
plt.text(20, 0.01, f'$Y_N$: a =  {round(y_N_fit[0],5)}\n      b = {round(y_N_fit[1],5)} \n'
                     f'$Y_U$: a = {round(y_U_fit[0],5)}\n      b = {round(y_U_fit[1],5)} \n'
                     f'H = {round(popt[0],2)} \n'
                     f'$T_m = {round(popt[1],2)}$K',
        verticalalignment='bottom', horizontalalignment='left', fontsize=12,  bbox= dict(boxstyle='round', alpha=0.1))
plt.legend()
plt.show()