# LABORATORIO 1

***Luz Marina Valderrama-5600741***

***Shesly Nicole Colorado - 5600756***

***Samuel Esteban Fonseca Luna - 5600808***

El siguiente codigo tiene la funcion de interpretar una serie de datos extraidos de la base de datos physionet. Los datos utilizados pertenecen a un electrocardiograma de estudio de apnea correspondiente (https://physionet.org/content/apnea-ecg/1.0.0/), los archivos utilizados con el siguiente codigo explicado, estan en este repositorio.

  
```
 import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
```

# Funciones para cargar datos
```
def cargarDat(nombre):
    """Carga archivo .dat con la señal ECG"""
    with open(nombre, 'rb') as f:
        return np.fromfile(f, dtype=np.int16)

def cargarHea(nombre):
    """Lee el archivo .hea y extrae parámetros de la señal"""
    with open(nombre, 'r') as f:
        lineas = f.readlines()

    fs = int(lineas[0].split()[2])  # Frecuencia de muestreo
    g = int(lineas[1].split()[2])   # Ganancia
    base = int(lineas[1].split()[2]) # Línea base
    return fs, g, base
```

El código define dos funciones para la carga de datos. cargarDat(nombre) abre un archivo .dat y extrae la señal ECG en formato binario, devolviendo un arreglo de valores enteros de 16 bits. cargarHea(nombre) lee el archivo .hea asociado que se relaciona con el encabezado de los datos extraidos en el .dat, extrayendo información clave como la frecuencia de muestreo, la ganancia y la línea base de la señal.

# Cargar archivos
```
dat = "a05.dat"
hea = "a05.hea"

ecg = cargarDat(dat)
fs, g, base = cargarHea(hea)
```

Se especifican los nombres de los archivos .dat y .hea a utilizar. Luego, la función cargarDat se emplea para obtener la señal ECG, mientras que cargarHea proporciona la frecuencia de muestreo, la ganancia y la línea base.

# Convertir señal a mV y corregir línea base
```
ecgMv = (ecg - base) / g
t = np.arange(len(ecgMv)) / fs

```
Para convertir la señal ECG a milivoltios (mV), se resta la línea base y se divide por la ganancia. Posteriormente, se genera un vector de tiempo t en segundos, calculado en función de la cantidad de muestras y la frecuencia de muestreo.

# Graficar señal ECG
```
plt.figure(figsize=(20, 7))
plt.plot(t, ecgMv, label="ECG (mV)", color="black")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Señal ECG")
plt.legend()
plt.grid()
plt.xlim([1000, (t[-1] * 0.0004) + 1000])
plt.show()

```
![Image](https://github.com/user-attachments/assets/1e572a45-86c1-4ae6-934a-4bb263c7f919)

Se visualiza la señal ECG mediante una gráfica donde el eje X representa el tiempo en segundos y el eje Y la amplitud en milivoltios. Se agregan etiquetas, una leyenda y una cuadrícula para mejorar la interpretación de los datos.

# Cálculo de estadísticos manualmente
```
mediaMan = 0
for x in ecgMv:
  mediaMan += x
  mmanual = mediaMan / len(ecgMv)
  
sum_sq_diff = 0
for x in ecgMv:
  sum_sq_diff += (x - mmanual) ** 2
  stdMan = (sum_sq_diff / len(ecgMv)) ** 0.5
  cvMan = stdMan / mmanual
```

Se calculan estadísticas básicas de la señal ECG sin usar funciones de librerías externas. La media se obtiene dividiendo la suma de los valores por la cantidad total de datos. La desviación estándar se determina calculando la diferencia cuadrática respecto a la media y dividiendo por el número total de muestras, extrayendo después la raíz cuadrada. Finalmente, se obtiene el coeficiente de variación dividiendo la desviación estándar entre la media.

# Estadísticos con funciones
```
media = np.mean(ecgMv)
std = np.std(ecgMv)
cv = std / media
```

Para validar los cálculos manuales, se utilizan funciones de numpy que realizan estos mismos cálculos de manera más eficiente. Se obtienen la media, la desviación estándar y el coeficiente de variación de la señal ECG.

# Histograma
```
plt.figure(figsize=(10, 5))
plt.hist(ecgMv, bins=50, color='blue', alpha=0.7, density=True, label="Histograma")
xVals = np.linspace(min(ecgMv), max(ecgMv), 100)
plt.plot(xVals, stats.norm.pdf(xVals, media, std), 'r-', label="Distribución Normal")
plt.xlabel("Amp (mV)")
plt.ylabel("Densidad")
plt.title("Histograma de ECG")
plt.legend()
plt.grid()
plt.show()
```

![Image](https://github.com/user-attachments/assets/44566bb4-b589-458d-9df2-bde965bdde67)

Se genera un histograma para visualizar la distribución de los valores de la señal ECG. Sobre este histograma se superpone una curva de distribución normal basada en la media y la desviación estándar calculadas previamente, lo que permite evaluar si la señal sigue una distribución gaussiana.

# Función de Probabilidad Acumulada (CDF)
```
ecgOrd = np.sort(ecgMv)
cdf = np.arange(len(ecgOrd)) / len(ecgOrd)

plt.figure(figsize=(10, 5))
plt.plot(ecgOrd, cdf, label="CDF", color='green')
plt.xlabel("Amp (mV)")
plt.ylabel("Probabilidad Acumulada")
plt.title("CDF de ECG")
plt.legend()
plt.grid()
plt.show()
```

![Image](https://github.com/user-attachments/assets/34d1867e-71f4-45a9-99dd-302a0018b2a0)

Se calcula la Función de Distribución Acumulada (CDF), ordenando los valores de la señal ECG de menor a mayor y asignando probabilidades acumuladas. Luego, se grafica para analizar la distribución de los datos y su comportamiento.

# Función para añadir ruido

El SNR (del inglés Signal-to-Noise Ratio) es una medida que compara el nivel de una señal con el nivel de ruido de fondo en un sistema. Se expresa como la relación entre la potencia de la señal y la potencia del ruido, y se mide en decibelios (dB).
### Importancia del SNR:
- Un SNR alto indica que la señal es más fuerte en comparación con el ruido, lo que generalmente resulta en una mejor calidad de la señal.

- Un SNR bajo significa que el ruido es más prominente, lo que puede degradar la calidad de la señal y dificultar su interpretación o procesamiento.

#### Fórmula:
SNR (dB)

![image](https://github.com/user-attachments/assets/da5257f5-beef-4639-ba65-ced1b4d0bfb2)

Donde 

_Pseñal_ es la potencia de la señal.
 
_Pruido_ es la potencia del ruido.

```
def ruido(ecg, tipo='gauss', snrDb=10):
    """Añade ruido a la señal y calcula el SNR"""
    potSig = np.mean(ecg ** 2)

    if tipo == 'gauss':
        potRuido = potSig / (10 ** (snrDb / 10))
        ruido = np.random.normal(0, np.sqrt(potRuido), len(ecg))
    elif tipo == 'impulso':
        ruido = np.random.choice([-1, 1], size=len(ecg)) * (potSig / 2)
    elif tipo == 'artefacto':
        ruido = np.sin(2 * np.pi * 60 * t) * (potSig / 10)
    else:
        raise ValueError("Tipo de ruido no válido")

    ecgRuido = ecg + ruido
    return ecgRuido, 10 * np.log10(potSig / np.mean(ruido ** 2))
```

Se define la función ruido(ecg, tipo, snrDb), que añade distintos tipos de ruido a la señal ECG. Si el ruido es gaussiano, se genera un ruido aleatorio con una potencia determinada por la relación señal-ruido (SNR). Si es de tipo impulso, se agregan valores aleatorios positivos y negativos con alta amplitud. Si es un artefacto, se suma una señal sinusoidal de 60 Hz simulando interferencia eléctrica.

# Añadir ruidos a la señal
```
ecgGauss, snrGauss = ruido(ecgMv, 'gauss', 10)
ecgImp, snrImp = ruido(ecgMv, 'impulso', 10)
ecgArt, snrArt = ruido(ecgMv, 'artefacto', 10)
```

Se generan tres versiones modificadas de la señal ECG, cada una con un tipo diferente de ruido: gaussiano, impulsivo y artefacto. Además, se calcula el SNR resultante de cada señal ruidosa para evaluar su calidad.


# Grafica individual de ruido 
```
plt.figure(figsize=(10, 5))
plt.plot(t, ecgGauss, label=f"Ruido Gauss (SNR = {snrGauss:.2f} dB)", color="r")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Gauss")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])
plt.show()
```

![Image](https://github.com/user-attachments/assets/4111ef56-4403-4188-8f62-ea823f5a4782)

```
plt.figure(figsize=(10, 5))
plt.plot(t, ecgImp, label=f"Ruido Impulso (SNR = {snrImp:.2f} dB)", color="g")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Impulso")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])
plt.show()
```

![Image](https://github.com/user-attachments/assets/fa40ba49-01bd-48c6-82a5-53b38c33a30c)

```
plt.figure(figsize=(10, 5))
plt.plot(t, ecgArt, label=f"Ruido Artefacto (SNR = {snrArt:.2f} dB)", color="b")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Artefacto")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])
plt.tight_layout()
plt.show()

```

![Image](https://github.com/user-attachments/assets/def59134-85f2-42d0-a78d-6616ab972a98)

# Graficar señales con ruido
```
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(t, ecgGauss, label=f"Ruido Gauss (SNR = {snrGauss:.2f} dB)", color="r")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Gauss")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])



plt.subplot(1, 3, 2)
plt.plot(t, ecgImp, label=f"Ruido Impulso (SNR = {snrImp:.2f} dB)", color="g")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Impulso")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])



plt.subplot(1, 3, 3)
plt.plot(t, ecgArt, label=f"Ruido Artefacto (SNR = {snrArt:.2f} dB)", color="b")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amp (mV)")
plt.title("Ruido Artefacto")
plt.grid()
plt.xlim([1000, (t[-1] * 0.00009) + 1000])

plt.tight_layout()
plt.show()

```


## Los tres tipos de ruidos

![Image](https://github.com/user-attachments/assets/0a60a863-47e8-4499-8cc4-b55f8b574231)

Se crean tres subgráficas en las que se muestran las versiones de la señal ECG con ruido gaussiano, impulsivo y de artefacto. Cada gráfico indica el tipo de ruido aplicado y el SNR obtenido.

# Mostrar resultados
```
print("* Estadísticos de ECG:")
print(f"* Media manual: {mmanual:.4f} mV, con función: {media:.4f} mV")
print(f"* Desviación estándar manual: {stdMan:.4f} mV, con función: {std:.4f} mV")
print(f"* Coeficiente de Variación manual: {cvMan:.4f}, con función: {cv:.4f}")
print(f"SNR con ruido Gauss: {snrGauss:.2f} dB")
print(f"SNR con ruido Impulso: {snrImp:.2f} dB")
print(f"SNR con ruido Artefacto: {snrArt:.2f} dB")
```

![image](https://github.com/user-attachments/assets/1522102f-3fbd-484e-ad14-edc54aaadaeb)


Se imprimen en pantalla los valores de la media, la desviación estándar y el coeficiente de variación calculados manualmente y con funciones de numpy. Además, se muestran los valores del SNR para cada tipo de ruido añadido, permitiendo comparar la relación señal-ruido en cada caso.
# Conclusiones
Como conclusiones encontramos 
1. La media y la desviación estándar permiten caracterizar la señal ECG en términos de amplitud y variabilidad.
2. El SNR nos indica qué tanto ruido afecta la señal; el ruido de artefacto fue el mas dificil, ya que su frecuencia es similar a la de la señal ECG.
3. El uso de histogramas y la CDF nos da una visión clara de la distribución de la amplitud de la señal.


 

