from flask import Flask, render_template, request, redirect, flash
from main import capture_and_save_images, mark_attendance
import os

app = Flask(__name__)
app.secret_key = "secretkey"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name'].strip()
    if name == "":
        flash("Please enter a name!")
        return redirect('/')
    capture_and_save_images(name)
    flash(f"{name} registered successfully!")
    return redirect('/')

@app.route('/mark')
def mark():
    mark_attendance()
    flash("Attendance process complete. Check CSV.")
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)