from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
@app.route('/homepage', methods=['GET', 'POST'])
def homepage():

    recommend = 0
    loaded_model = pickle.load(open("model/book_recommender.pkl", "rb"))
    images = pd.read_csv('datasets/images.csv')
    book_pivot = pd.read_csv('datasets/book_pivot.csv')
    book_pivot.set_index('title', inplace=True)
    book_names = list(book_pivot.index)


    if request.method == 'POST':
        Id = int(request.form['book'])
        distances, suggestions = loaded_model.kneighbors(
            book_pivot.iloc[Id, :].values.reshape(1, -1))
        suggestions = suggestions[0]
        authors = []
        years = []
        publishers = []
        titles = []
        isbn_no = []
        recommend = 1
        choice = []
        

        name = book_names[Id]
        choice.append(name)
        choice.append(images[images['title'] == name]['author'].values[0])
        choice.append(images[images['title'] == name]['year'].values[0])
        choice.append(images[images['title'] == name]['publisher'].values[0])
        choice.append(images[images['title'] == name]['ISBN'].values[0])
        
        for i in range(len(suggestions)-1):
            name = book_pivot.index[suggestions[i+1]]
            author = images[images['title'] == name]['author'].values[0]
            yr = images[images['title'] == name]['year'].values[0]
            publish = images[images['title'] == name]['publisher'].values[0]
            isbn = images[images['title'] == name]['ISBN'].values[0]

            authors.append(author)
            years.append(yr)
            publishers.append(publish)
            titles.append(name)
            isbn_no.append(isbn)
        return render_template('homepage.html', book_names=book_names,choice=choice, titles=titles, author=authors, year=years, publisher=publishers, recommend=recommend, isbn_no=isbn_no)

    return render_template('homepage.html', book_names=book_names)


if __name__ == '__main__':
    app.run(debug=True)
