import sys
import os
import tkinter as tk
from tkinter import messagebox

# Agregar el directorio src al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    """
    Punto de entrada principal de la aplicación
    """
    try:
        # Importar la interfaz principal
        from gui.main_interface import DigitsClassifierApp
        
        # Crear ventana principal
        root = tk.Tk()
        
        # Inicializar aplicación
        app = DigitsClassifierApp(root)
        
        # Iniciar loop de la interfaz
        root.mainloop()
        
    except ImportError as e:
        # Crear ventana de error básica si falla la importación
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana principal
        messagebox.showerror(
            "Error de Importación", 
            f"No se pudo importar la interfaz principal:\n{str(e)}\n\n"
            "Verifica que todos los archivos estén en su lugar correcto."
        )
        root.destroy()
        
    except Exception as e:
        # Manejar otros errores
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error", 
            f"Error inesperado:\n{str(e)}"
        )
        root.destroy()

if __name__ == "__main__":
    print("Iniciando Clasificador de Dígitos")
    main()