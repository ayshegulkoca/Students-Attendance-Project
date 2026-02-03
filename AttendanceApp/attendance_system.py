import os
import sys
import csv
import shutil
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from openpyxl import Workbook

# ============================================================
# PATHS
# ============================================================
def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_dir()
os.chdir(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_FILE = os.path.join(BASE_DIR, "trainer.yml")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.csv")

FACE_SIZE = (200, 200)
SAMPLES_PER_STUDENT = 40
THRESH = 65

CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# ============================================================
# CAMERA
# ============================================================
def open_camera():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        for i in range(5):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
                cap.release()
    return None

# ============================================================
# FILE HELPERS
# ============================================================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_students_header():
    if not os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["label", "StudentID", "Name"])


def create_new_attendance_file():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(BASE_DIR, f"attendance_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["StudentID", "Name", "Date", "Time"])
    return path


def load_students():
    ensure_students_header()
    data = {}
    with open(STUDENTS_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            data[int(r["label"])] = {"StudentID": r["StudentID"], "Name": r["Name"]}
    return data


def next_label(existing):
    n = 1
    while n in existing:
        n += 1
    return n

# ============================================================
# FACE MODEL
# ============================================================
def get_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


def train_model():
    faces, labels = [], []
    for lbl in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, lbl)
        if not os.path.isdir(folder):
            continue
        for img in os.listdir(folder):
            g = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            faces.append(cv2.resize(g, FACE_SIZE))
            labels.append(int(lbl))

    if not faces:
        return False

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels))
    rec.write(TRAINER_FILE)
    return True


def load_recognizer():
    if not os.path.exists(TRAINER_FILE):
        return None
    r = cv2.face.LBPHFaceRecognizer_create()
    r.read(TRAINER_FILE)
    return r

# ============================================================
# REGISTER STUDENT
# ============================================================
def register_student():
    name = name_entry.get().strip()
    sid = id_entry.get().strip()

    if not name or not sid:
        messagebox.showerror("Error", "Enter Student Name and ID")
        return

    ensure_dirs()
    students = load_students()
    label = next_label(students.keys())

    folder = os.path.join(DATA_DIR, str(label))
    os.makedirs(folder, exist_ok=True)

    cap = open_camera()
    if cap is None:
        messagebox.showerror("Camera Error", "Camera not accessible")
        shutil.rmtree(folder)
        return

    detector = get_face_detector()
    count = 0

    while count < SAMPLES_PER_STUDENT:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        cv2.imshow("Register - Press ESC to stop", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        shutil.rmtree(folder)
        messagebox.showerror("Error", "No face captured")
        return

    with open(STUDENTS_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([label, sid, name])

    train_model()
    messagebox.showinfo("Success", "Student registered successfully")

# ============================================================
# TAKE ATTENDANCE  (SAVE BUTTON FIXED)
# ============================================================
def take_attendance():
    students = load_students()
    rec = load_recognizer()

    if rec is None:
        messagebox.showerror("Error", "No trained model found")
        return

    cap = open_camera()
    if cap is None:
        messagebox.showerror("Camera Error", "Camera not accessible")
        return

    session_file = create_new_attendance_file()
    seen = set()
    detector = get_face_detector()

    win = tk.Toplevel(root)
    win.title("Take Attendance — RECORDING")
    win.geometry("1100x820")
    win.configure(bg="#0b0b10")

    # ===== RECORDING LABEL =====
    tk.Label(
        win,
        text="● Recording Attendance",
        fg="#a855f7",
        bg="#0b0b10",
        font=("Segoe UI", 12, "bold")
    ).pack(pady=8)

    # ===== CAMERA =====
    cam_frame = tk.Frame(win, bg="#a855f7", padx=3, pady=3)
    cam_frame.pack()

    cam_label = tk.Label(cam_frame, bg="black")
    cam_label.pack()

    # ===== TABLE =====
    table_frame = tk.Frame(win, bg="#0b0b10")
    table_frame.pack(fill="x", padx=16, pady=10)

    cols = ("StudentID", "Name", "Date", "Time")
    tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=5)

    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=250, anchor="center")

    tree.pack(fill="x")

    # ===== SAVE BAR (ALWAYS VISIBLE) =====
    save_bar = tk.Frame(win, bg="#0f0f14", height=80)
    save_bar.pack(side="bottom", fill="x")

    def save_excel():
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")]
        )
        if not path:
            return

        wb = Workbook()
        ws = wb.active
        with open(session_file) as f:
            for row in csv.reader(f):
                ws.append(row)
        wb.save(path)
        messagebox.showinfo("Saved", "Attendance exported to Excel")

    tk.Button(
        save_bar,
        text="Save / Export Attendance to Excel",
        command=save_excel,
        bg="#1a1a1a",
        fg="white",
        font=("Segoe UI", 12, "bold"),
        width=32,
        pady=12
    ).pack(pady=16)

    def refresh():
        for i in tree.get_children():
            tree.delete(i)
        with open(session_file) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                tree.insert("", tk.END, values=row)

    def mark(sid, name):
        if sid in seen:
            return
        now = datetime.now()
        with open(session_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([sid, name, now.date(), now.strftime("%H:%M:%S")])
        seen.add(sid)
        refresh()

    def update():
        ret, frame = cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            lbl, conf = rec.predict(face)

            if conf <= THRESH and lbl in students:
                s = students[lbl]
                mark(s["StudentID"], s["Name"])
                txt = s["Name"]
            else:
                txt = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (168, 85, 247), 2)
            cv2.putText(frame, txt, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (168, 85, 247), 2)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cam_label.imgtk = img
        cam_label.configure(image=img)
        win.after(10, update)

    def close():
        cap.release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", close)
    refresh()
    update()

# ============================================================
# MAIN UI
# ============================================================
root = tk.Tk()
root.title("Attendance System")
root.geometry("440x560")
root.configure(bg="#0f0f14")
root.resizable(False, False)

tk.Label(
    root,
    text="Attendance System",
    font=("Segoe UI", 20, "bold"),
    fg="white",
    bg="#0f0f14"
).pack(pady=20)

form = tk.Frame(root, bg="#0f0f14")
form.pack(pady=10)

tk.Label(form, text="Student Name", fg="white", bg="#0f0f14").pack(anchor="w")
name_entry = tk.Entry(form, width=30)
name_entry.pack(pady=6)

tk.Label(form, text="Student ID", fg="white", bg="#0f0f14").pack(anchor="w")
id_entry = tk.Entry(form, width=30)
id_entry.pack(pady=6)

def dark_button(text, cmd):
    return tk.Button(
        root,
        text=text,
        command=cmd,
        bg="#1a1a1a",
        fg="white",
        font=("Segoe UI", 12, "bold"),
        width=26,
        pady=12
    )

dark_button("Register Student", register_student).pack(pady=10)
dark_button("Take Attendance", take_attendance).pack(pady=10)
dark_button("Exit", root.destroy).pack(pady=20)

ensure_dirs()
ensure_students_header()

root.mainloop()
