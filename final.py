"""
Disease Detection System  -  interactive Tkinter front end
Run:  python final.py     (keep model.joblib and encoder.joblib in the same folder)

Features
  * Live search/filter over all symptoms
  * Click-to-remove "chips" for selected symptoms
  * Live prediction (updates as you tick) with animated top-5 confidence bars
  * Confidence level + hint when too few symptoms are selected
  * Prediction history (double-click an entry to restore those symptoms)
  * Export report, Clear all, Light/Dark theme
  * Keyboard shortcuts: Ctrl+F search | Ctrl+Enter predict | Ctrl+L clear | Esc clear search
"""
import os
import re
import datetime
import warnings
import tkinter as tk
from tkinter import messagebox, filedialog

import joblib
import numpy as np

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))

# Used only if the model has no stored feature names
FALLBACK_SYMPTOMS = [
 'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue',
 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite',
 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
 'red_sore_around_nose', 'yellow_crust_ooze'
]

LABEL_FIXES = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
}

THEMES = {
    "light": dict(bg="#eef2f7", card="#ffffff", text="#1f2937", muted="#6b7280", accent="#2563eb",
                  accent_fg="#ffffff", chip="#dbeafe", chip_fg="#1e40af", border="#d1d5db",
                  good="#16a34a", warn="#d97706", bad="#dc2626", bar_bg="#e5e7eb", header="#1e3a8a"),
    "dark":  dict(bg="#0f172a", card="#1e293b", text="#e5e7eb", muted="#94a3b8", accent="#3b82f6",
                  accent_fg="#ffffff", chip="#1e3a8a", chip_fg="#dbeafe", border="#334155",
                  good="#22c55e", warn="#f59e0b", bad="#ef4444", bar_bg="#334155", header="#020617"),
}


def pretty_symptom(name):
    dup = name.endswith(".1")
    text = re.sub(r"\s+", " ", name.replace(".1", "").replace("_", " ")).strip().capitalize()
    return text + " (2)" if dup else text


# ======================= Logic layer =======================
class DiseaseModel:
    def __init__(self):
        self.model = joblib.load(os.path.join(BASE, "model.joblib"))
        self.encoder = joblib.load(os.path.join(BASE, "encoder.joblib"))
        names = getattr(self.model, "feature_names_in_", None)
        self.symptoms = list(names) if names is not None else FALLBACK_SYMPTOMS
        self.index = {s: i for i, s in enumerate(self.symptoms)}
        raw = self.encoder.inverse_transform(self.model.classes_)
        self.classes = [LABEL_FIXES.get(c.strip(), re.sub(r"\s+", " ", c.strip())) for c in raw]

    def predict_top(self, selected, k=5):
        x = np.zeros((1, len(self.symptoms)))
        for s in selected:
            x[0, self.index[s]] = 1
        proba = self.model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1][:k]
        return [(self.classes[i], float(proba[i])) for i in order if proba[i] > 0]


# ======================= Custom widgets =======================
class Bar(tk.Canvas):
    """Horizontal progress bar with a smooth animation."""
    def __init__(self, master, **kw):
        super().__init__(master, height=12, highlightthickness=0, **kw)
        self.frac = self.target = 0.0
        self.color = "#2563eb"
        self._job = None
        self.rect = self.create_rectangle(0, 0, 0, 12, width=0)
        self.bind("<Configure>", lambda e: self.draw())

    def set(self, target, color):
        self.target, self.color = target, color
        if self._job:
            self.after_cancel(self._job)
        self.step()

    def step(self):
        diff = self.target - self.frac
        if abs(diff) < 0.004:
            self.frac = self.target
            self._job = None
        else:
            self.frac += diff * 0.25
            self._job = self.after(16, self.step)
        self.draw()

    def draw(self):
        self.coords(self.rect, 0, 0, self.winfo_width() * self.frac, 12)
        self.itemconfig(self.rect, fill=self.color)


# ======================= GUI =======================
class DiseaseApp:
    TOP_K = 5

    def __init__(self, root, engine):
        self.root, self.engine = root, engine
        self.theme = "light"
        self.T = THEMES[self.theme]
        self.vars = {s: tk.BooleanVar() for s in engine.symptoms}
        self.checks = {}
        self.history = []
        self.last_result = []
        self._themed = []          # (widget, {option: theme_key})
        self._predict_job = None
        self.live = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()

        root.title("Disease Detection System")
        root.geometry("1100x720")
        root.minsize(900, 600)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        self.build_header()
        self.build_body()
        self.build_status()
        self.bind_keys()
        self.apply_theme()
        self.refresh_all()

    # ---------- helpers ----------
    def reg(self, widget, **opts):
        self._themed.append((widget, opts))
        return widget

    def card(self, parent, title):
        frame = self.reg(tk.Frame(parent, highlightthickness=1), bg="card", highlightbackground="border",
                         highlightcolor="border")
        self.reg(tk.Label(frame, text=title, font=("Segoe UI", 11, "bold"), anchor="w"), bg="card", fg="text") \
            .pack(fill="x", padx=12, pady=(8, 4))
        return frame

    def button(self, parent, text, command, primary=False):
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0, padx=14, pady=7,
                      font=("Segoe UI", 10, "bold"), cursor="hand2")
        if primary:
            self.reg(b, bg="accent", fg="accent_fg", activebackground="accent", activeforeground="accent_fg")
        else:
            self.reg(b, bg="bar_bg", fg="text", activebackground="border", activeforeground="text")
        return b

    # ---------- layout ----------
    def build_header(self):
        header = self.reg(tk.Frame(self.root), bg="header")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        box = self.reg(tk.Frame(header), bg="header")
        box.grid(row=0, column=0, sticky="w", padx=20, pady=10)
        self.reg(tk.Label(box, text="🩺  Disease Detection System", font=("Segoe UI", 18, "bold")),
                 bg="header", fg="accent_fg").pack(anchor="w")
        self.reg(tk.Label(box, text="Pick your symptoms – predictions update live", font=("Segoe UI", 10)),
                 bg="header", fg="border").pack(anchor="w")
        self.theme_btn = self.button(header, "🌙 Dark mode", self.toggle_theme)
        self.theme_btn.grid(row=0, column=1, padx=20)

    def build_body(self):
        body = self.reg(tk.Frame(self.root), bg="bg")
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=12)
        body.columnconfigure(0, weight=3, uniform="c")
        body.columnconfigure(1, weight=2, uniform="c")
        body.rowconfigure(0, weight=1)
        self.build_left(body)
        self.build_right(body)

    def build_left(self, body):
        left = self.card(body, "1 · Select symptoms")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        bar = self.reg(tk.Frame(left), bg="card")
        bar.pack(fill="x", padx=12)
        self.reg(tk.Label(bar, text="🔍"), bg="card", fg="text").pack(side="left")
        self.search = self.reg(tk.Entry(bar, textvariable=self.search_var, relief="flat", font=("Segoe UI", 11),
                                        highlightthickness=1), bg="bg", fg="text", insertbackground="text",
                               highlightbackground="border", highlightcolor="accent")
        self.search.pack(side="left", fill="x", expand=True, padx=6, ipady=4)
        self.search_var.trace_add("write", lambda *a: self.apply_filter())
        self.show_lbl = self.reg(tk.Label(bar, font=("Segoe UI", 9)), bg="card", fg="muted")
        self.show_lbl.pack(side="right")

        wrap = self.reg(tk.Frame(left), bg="card")
        wrap.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas = self.reg(tk.Canvas(wrap, highlightthickness=0), bg="card")
        sb = tk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = self.reg(tk.Frame(self.canvas), bg="card")
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        for c in range(2):
            self.inner.columnconfigure(c, weight=1, uniform="s")
        for s in self.engine.symptoms:
            cb = self.reg(tk.Checkbutton(self.inner, text=pretty_symptom(s), variable=self.vars[s], anchor="w",
                                         font=("Segoe UI", 10), command=self.on_toggle, cursor="hand2",
                                         bd=0, highlightthickness=0),
                          bg="card", fg="text", activebackground="card", activeforeground="accent",
                          selectcolor="card")
            self.checks[s] = cb
        # mouse wheel (Windows / macOS / Linux)
        self.canvas.bind("<Enter>", lambda e: self.bind_wheel(True))
        self.canvas.bind("<Leave>", lambda e: self.bind_wheel(False))
        self.inner.bind("<Enter>", lambda e: self.bind_wheel(True))

    def build_right(self, body):
        right = self.reg(tk.Frame(body), bg="bg")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # selected chips
        sel = self.card(right, "2 · Your symptoms")
        sel.grid(row=0, column=0, sticky="ew")
        self.chips = self.reg(tk.Text(sel, height=4, wrap="word", relief="flat", cursor="arrow",
                                      font=("Segoe UI", 10), highlightthickness=0, state="disabled"),
                              bg="card", fg="muted")
        self.chips.pack(fill="x", padx=12, pady=(0, 10))

        # results
        res = self.card(right, "3 · Prediction")
        res.grid(row=1, column=0, sticky="nsew", pady=10)
        self.top_lbl = self.reg(tk.Label(res, text="—", font=("Segoe UI", 17, "bold"), anchor="w",
                                         wraplength=380, justify="left"), bg="card", fg="accent")
        self.top_lbl.pack(fill="x", padx=12)
        self.conf_lbl = self.reg(tk.Label(res, text="", font=("Segoe UI", 10), anchor="w"), bg="card", fg="muted")
        self.conf_lbl.pack(fill="x", padx=12, pady=(0, 6))
        grid = self.reg(tk.Frame(res), bg="card")
        grid.pack(fill="x", padx=12)
        grid.columnconfigure(1, weight=1)
        self.rows = []
        for r in range(self.TOP_K):
            name = self.reg(tk.Label(grid, anchor="w", font=("Segoe UI", 9), width=24), bg="card", fg="text")
            bar = self.reg(Bar(grid), bg="bar_bg")
            pct = self.reg(tk.Label(grid, width=6, anchor="e", font=("Segoe UI", 9, "bold")), bg="card", fg="text")
            name.grid(row=r, column=0, sticky="w", pady=3)
            bar.grid(row=r, column=1, sticky="ew", padx=6)
            pct.grid(row=r, column=2)
            self.rows.append((name, bar, pct))
        self.reg(tk.Label(res, text="⚠ Educational project – not a medical diagnosis. Consult a doctor.",
                          font=("Segoe UI", 8), anchor="w", wraplength=380, justify="left"),
                 bg="card", fg="muted").pack(fill="x", padx=12, pady=(8, 8), side="bottom")

        # history
        hist = self.card(right, "History  (double-click to restore)")
        hist.grid(row=2, column=0, sticky="ew")
        self.hist_box = self.reg(tk.Listbox(hist, height=4, relief="flat", font=("Segoe UI", 9),
                                            highlightthickness=0, activestyle="none"),
                                 bg="card", fg="text", selectbackground="accent", selectforeground="accent_fg")
        self.hist_box.pack(fill="x", padx=12, pady=(0, 10))
        self.hist_box.bind("<Double-Button-1>", self.restore_history)

        # buttons
        btns = self.reg(tk.Frame(right), bg="bg")
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.button(btns, "Predict", self.predict_now, primary=True).pack(side="left")
        self.button(btns, "Clear all", self.clear_all).pack(side="left", padx=6)
        self.button(btns, "Export report", self.export).pack(side="left")
        self.reg(tk.Checkbutton(btns, text="Live", variable=self.live, command=self.on_toggle,
                                font=("Segoe UI", 10), bd=0, highlightthickness=0),
                 bg="bg", fg="text", activebackground="bg", activeforeground="text",
                 selectcolor="card").pack(side="right")

    def build_status(self):
        self.status = self.reg(tk.Label(self.root, anchor="w", font=("Segoe UI", 9), padx=14, pady=4),
                               bg="card", fg="muted")
        self.status.grid(row=2, column=0, sticky="ew")

    def bind_keys(self):
        r = self.root
        r.bind("<Control-f>", lambda e: (self.search.focus_set(), self.search.select_range(0, "end")))
        r.bind("<Control-Return>", lambda e: self.predict_now())
        r.bind("<Control-l>", lambda e: self.clear_all())
        self.search.bind("<Escape>", lambda e: self.search_var.set(""))

    # ---------- theme ----------
    def apply_theme(self):
        self.T = THEMES[self.theme]
        self.root.configure(bg=self.T["bg"])
        for widget, opts in self._themed:
            widget.configure(**{k: self.T[v] for k, v in opts.items()})
        self.theme_btn.configure(text="☀ Light mode" if self.theme == "dark" else "🌙 Dark mode")
        self.refresh_all()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()

    # ---------- mouse wheel ----------
    def bind_wheel(self, on):
        if on:
            self.root.bind_all("<MouseWheel>", self._wheel)
            self.root.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
            self.root.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))
        else:
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.root.unbind_all(ev)

    def _wheel(self, e):
        step = -1 * (e.delta // 120) if abs(e.delta) >= 120 else -e.delta   # Windows vs macOS
        self.canvas.yview_scroll(step * 2, "units")

    # ---------- behaviour ----------
    def selected(self):
        return [s for s, v in self.vars.items() if v.get()]

    def apply_filter(self):
        q = self.search_var.get().strip().lower().replace(" ", "_")
        for cb in self.checks.values():
            cb.grid_remove()
        shown = [s for s in self.engine.symptoms if q in s.lower().replace(" ", "_")]
        for i, s in enumerate(shown):
            self.checks[s].grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=1)
        self.show_lbl.configure(text=f"Showing {len(shown)}/{len(self.engine.symptoms)}")
        self.canvas.yview_moveto(0)

    def on_toggle(self):
        self.refresh_chips()
        self.update_status()
        if self._predict_job:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None
        if self.live.get():
            self._predict_job = self.root.after(150, self.predict_live)
        else:
            self.show_waiting()

    def refresh_all(self):
        self.apply_filter()
        self.refresh_chips()
        self.update_status()
        if self.last_result:
            self.render(self.last_result, len(self.selected()))
        else:
            self.show_waiting()

    def refresh_chips(self):
        sel = self.selected()
        self.chips.configure(state="normal")
        self.chips.delete("1.0", "end")
        if not sel:
            self.chips.insert("end", "Nothing selected yet. Tick symptoms on the left or use the search box.")
        for s in sel:
            chip = tk.Label(self.chips, text=f" {pretty_symptom(s)}  ✕ ", font=("Segoe UI", 9),
                            bg=self.T["chip"], fg=self.T["chip_fg"], cursor="hand2", pady=2)
            chip.bind("<Button-1>", lambda e, sym=s: self.remove_symptom(sym))
            self.chips.window_create("end", window=chip, padx=2, pady=2)
        self.chips.configure(state="disabled")

    def remove_symptom(self, s):
        self.vars[s].set(False)
        self.on_toggle()

    def update_status(self):
        n = len(self.selected())
        self.status.configure(text=f"Selected: {n}   |   Model: {type(self.engine.model).__name__}   |   "
                                   f"{len(self.engine.classes)} diseases, {len(self.engine.symptoms)} symptoms   |   "
                                   f"Ctrl+F search · Ctrl+Enter predict · Ctrl+L clear")

    def show_waiting(self):
        self.last_result = []
        self.top_lbl.configure(text="—", fg=self.T["muted"])
        self.conf_lbl.configure(text="Select symptoms and press Predict." if not self.live.get()
                                else "Select symptoms to see a live prediction.")
        for name, bar, pct in self.rows:
            name.grid_remove(); bar.grid_remove(); pct.grid_remove()

    def predict_live(self):
        self._predict_job = None
        sel = self.selected()
        if not sel:
            self.show_waiting()
            return
        self.render(self.engine.predict_top(sel, self.TOP_K), len(sel))

    def predict_now(self):
        sel = self.selected()
        if not sel:
            messagebox.showwarning("Warning", "Please select at least one symptom.")
            return
        result = self.engine.predict_top(sel, self.TOP_K)
        self.render(result, len(sel))
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.history.insert(0, dict(time=stamp, symptoms=sel, result=result))
        self.hist_box.insert(0, f"{stamp}  {result[0][0]} ({result[0][1]*100:.0f}%) · {len(sel)} symptoms")

    def render(self, result, n_sel):
        self.last_result = result
        if not result:
            return
        T = self.T
        disease, p = result[0]
        # RandomForest (max_depth=10) probabilities are spread out: median on real rows is ~0.34
        level, color = ("High", T["good"]) if p >= .45 else ("Medium", T["warn"]) if p >= .25 else ("Low", T["bad"])
        hint = "  ·  add more symptoms for a sharper result" if n_sel < 3 else ""
        self.top_lbl.configure(text=disease, fg=T["accent"])
        self.conf_lbl.configure(text=f"{level} confidence ({p*100:.1f}%){hint}", fg=color)
        for i, (name, bar, pct) in enumerate(self.rows):
            if i < len(result):
                d, pr = result[i]
                name.grid(); bar.grid(); pct.grid()
                name.configure(text=d)
                pct.configure(text=f"{pr*100:.1f}%")
                bar.set(pr, T["accent"] if i == 0 else T["muted"])
            else:
                name.grid_remove(); bar.grid_remove(); pct.grid_remove()

    def clear_all(self):
        for v in self.vars.values():
            v.set(False)
        self.search_var.set("")
        self.on_toggle()

    def restore_history(self, _):
        idx = self.hist_box.curselection()
        if not idx:
            return
        entry = self.history[idx[0]]
        for s, v in self.vars.items():
            v.set(s in entry["symptoms"])
        self.on_toggle()

    def export(self):
        if not self.last_result:
            messagebox.showinfo("Export", "Run a prediction first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="prediction_report.txt",
                                            filetypes=[("Text file", "*.txt")])
        if not path:
            return
        lines = ["DISEASE DETECTION REPORT",
                 f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "",
                 "Symptoms:"] + [f"  - {pretty_symptom(s)}" for s in self.selected()] + ["", "Top predictions:"]
        lines += [f"  {i}. {d}  ({p*100:.1f}%)" for i, (d, p) in enumerate(self.last_result, 1)]
        lines += ["", "Disclaimer: educational project, not a medical diagnosis."]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("Export", f"Report saved:\n{path}")


if __name__ == "__main__":
    try:
        engine = DiseaseModel()
    except FileNotFoundError as err:
        tk.Tk().withdraw()
        messagebox.showerror("Missing file", f"Could not load model files:\n{err}")
        raise SystemExit(1)
    root = tk.Tk()
    DiseaseApp(root, engine)
    root.mainloop()