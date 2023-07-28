from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from csv import writer

from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers

from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


UPLOAD_FOLDER = r'E:\Cardless ATM\static\uploads'



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'

# Enter your database connection details below
batch_size = 32
img_height = 255
img_width = 255
# Enter your database connection details below

app.config["MYSQL_HOST"]='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='root'
app.config['MYSQL_DB']='pythonlogin'
# Intialize MySQL
mysql = MySQL(app)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
class_names = ['dhanush', 'kowshik', 'mahesh', 'manjunath', 'parvatha', 'rao']
encoded_string1=''
encoded_string2=''

img_height1 = 255
img_width1 = 255
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    msg = ''
    print("login")
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        # Fetch one record and return result
        # cursor=mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('Select * from accounts where username=%s AND password=%s',(username,password))
        # account = cursor.fetchone()
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['username'] = "admin"
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        
        # User is loggedin show them the home page
        return render_template('index.html')
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
@app.route('/pythonlogin/about')
def about():
    # Check if user is loggedin
    # User is not loggedin redirect to login page
    return render_template('about.html')
@app.route('/pythonlogin/predication')
def predication():
    # Check if user is loggedin
    # User is not loggedin redirect to login page
    return render_template('predication.html')
def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route("/pythonlogin/predication1",methods=['GET', 'POST'])
def index1():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        file1 = request.files['file1']
        if file.filename == '' or file1.filename == '':
            print('No file selected')
            return redirect(request.url)
        if (file and check_allowed_file(file.filename)) and (file1 and check_allowed_file(file1.filename)):
            filename = secure_filename(file.filename)
            print(filename)
            img = Image.open(file.stream)
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string1 = base64.b64encode(image_bytes).decode()   



            filename1 = secure_filename(file1.filename)
            print(filename1)
            img1 = Image.open(file1.stream)
            with BytesIO() as buf:
                img1.save(buf, 'jpeg')
                image_bytes1 = buf.getvalue()
            encoded_string2 = base64.b64encode(image_bytes1).decode()  
        return render_template('predication.html', filename=filename,filename1=filename1,img_data1=encoded_string1, img_data2=encoded_string2), 200
    else:
        return render_template('predication.html', img_data1=""), 200

@app.route('/pythonlogin/upload_image', methods=['POST'])
def upload_image():
    if 'loggedin' in session:
        if request.method == 'POST':
            filename = request.form['file']
            filename1 = request.form['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            print(path)
            print(path1)
            num_classes=6
            model = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])
            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

            model.load_weights("Cardless_face_Model.h5")


            test_data_path = path

            img = keras.preprocessing.image.load_img(
                test_data_path, target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_name = class_names[np.argmax(score)]



            model1 = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
            ])

            model1.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

            model1.load_weights("Cardless_finger_Model.h5")
            img1 = keras.preprocessing.image.load_img(path1, target_size=(img_height1, img_width1))
            img_array1 = keras.preprocessing.image.img_to_array(img1)
            img_array1 = tf.expand_dims(img_array1, 0) # Create a batch
            predictions1 = model1.predict(img_array1)
            score1 = tf.nn.softmax(predictions1[0])
            class_name1 = class_names[np.argmax(score1)]

            print(class_name, class_name1)
            msg="Face and finger dosn't match"

            if class_name==class_name1:
                return render_template('atm.html')
            else:
                return render_template('predication.html',msg=msg)

        
        else:
            return redirect(request.url)
    return redirect(url_for('login'))
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
       
        
        
                
            
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        
        account = cursor.fetchone()
        cursor1 = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor1.execute('SELECT * FROM predictiondetails WHERE username = %s', (session['username'],))
        prediction_details = cursor1.fetchall()
        # Show the profile page with account info
        return render_template('profile.html', account=account,prediction_details = prediction_details)
    # User is not loggedin redirect to login page
    return redirect(url_for('login')) 
    



    





if __name__ =='__main__':
	app.run()
