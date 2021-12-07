
from tensorflow.keras.models import load_model 
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request



app = Flask(__name__)

dic = {0: 'baseball',
 1: 'basketball',
 2: 'beachballs',
 3: 'billiard ball',
 4: 'bowling ball',
 5: 'brass',
 6: 'buckeyballs',
 7: 'cannon ball',
 8: 'cricket ball',
 9: 'eyeballs',
 10: 'football',
 11: 'golf ball',
 12: 'marble',
 13: 'meat ball',
 14: 'medicine ball',
 15: 'paint balls',
 16: 'puffballs',
 17: 'screwballs',
 18: 'soccer ball',
 19: 'tennis ball',
 20: 'volley ball',
 21: 'water polo ball',
 22: 'wiffle ball',
 23: 'wrecking ball'}


model = load_model('model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    #p = model.predict(i)
    p = model.predict(i) 
    print(p.astype(int))
    p = p.argmax(axis=-1)
    print(p)
    return dic[p[0]]



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename    
        img.save(img_path)
        p = predict_label(img_path)
        
        

    return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
    #app.debug = True
    app.run(debug = False)