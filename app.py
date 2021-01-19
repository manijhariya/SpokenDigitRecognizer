from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import PREDICT_MODEL as PM
import os

pm = PM.PREDICT_MODEL()

FileUploaded = 0
UPLOAD_FOLDER = 'UserImages/'
ALLOWED_EXTENSIONS = {'wav','wave'}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['GET','POST'])
def index():
	if  request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
		file = request.files['file']
		if file.filename == '':
			flash('No Selected file')
			return redirect(request.url)
		if file  and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filename = str(FileUploaded) + filename
			filename = os.path.join(app.config['UPLOAD_FOLDER'],filename)
			file.save(filename)
			FileUploaded += 1
			result = pm.predict(filename)
			return render_template('index.html',data=result)

	return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)
