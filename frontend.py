import tkinter as tk
from tkinter import ttk


class RotaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retail Rota Generator")
        self.root.geometry("700x500")

        # Title
        title = ttk.Label(
            root,
            text="Retail Rota Generator",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=20)

        # Information
        info = ttk.Label(
            root,
            text="Welcome!\n\nThis application will generate a staff rota from an Excel file.",
            justify="center"
        )
        info.pack(pady=20)

        # Generate Button
        generate_button = ttk.Button(
            root,
            text="Generate Rota",
            command=self.generate_rota
        )
        generate_button.pack(pady=20)

    def generate_rota(self):
        print("Generate button pressed.")


def run_app():
    root = tk.Tk()
    app = RotaApp(root)
    root.mainloop()