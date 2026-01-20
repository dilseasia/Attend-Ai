import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class DBViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SQLite DB Viewer")
        self.geometry("800x500")
        self.db_path = ""
        self.conn = None
        self.cursor = None

        # UI Elements
        self.create_widgets()

    def create_widgets(self):
        # Top frame for DB selection
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Database:").pack(side=tk.LEFT)
        self.db_entry = tk.Entry(top_frame, width=60)
        self.db_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_db).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Connect", command=self.connect_db).pack(side=tk.LEFT, padx=5)

        # Table list
        self.table_listbox = tk.Listbox(self, width=30)
        self.table_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        self.table_listbox.bind("<<ListboxSelect>>", self.show_table_content)

        # Table content
        self.tree = ttk.Treeview(self)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

    def browse_db(self):
        path = filedialog.askopenfilename(filetypes=[("SQLite DB", "*.db *.sqlite3 *.sqlite"), ("All files", "*.*")])
        if path:
            self.db_entry.delete(0, tk.END)
            self.db_entry.insert(0, path)

    def connect_db(self):
        self.db_path = self.db_entry.get()
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.table_listbox.delete(0, tk.END)
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in self.cursor.fetchall()]
            for table in tables:
                self.table_listbox.insert(tk.END, table)
            messagebox.showinfo("Connected", f"Connected to {self.db_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not connect to database:\n{e}")

    def show_table_content(self, event):
        selection = self.table_listbox.curselection()
        if not selection:
            return
        table = self.table_listbox.get(selection[0])
        try:
            self.cursor.execute(f"PRAGMA table_info({table});")
            columns = [info[1] for info in self.cursor.fetchall()]
            self.cursor.execute(f"SELECT * FROM {table};")
            rows = self.cursor.fetchall()

            # Clear previous tree
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = columns
            self.tree["show"] = "headings"
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor=tk.W)
            for row in rows:
                self.tree.insert("", tk.END, values=row)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read table:\n{e}")

if __name__ == "__main__":
    app = DBViewer()
    app.mainloop()